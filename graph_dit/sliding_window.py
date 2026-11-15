"""Continue a frozen CylinderFlow joint64 predictor with overlapping windows."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch

from .evaluate import Predictor, load_selected
from .metrics import (
    compute_metrics,
    node_area_weights,
    triangle_vorticity_divergence,
)
from .performance import configure, runtime_identity, sync
from .runtime import write_json


WINDOW = 64
DT = 0.08


def window_starts(future_frames: int, stride: int) -> list[int]:
    """Use complete model windows, trimming only the final saved trajectory."""
    starts = [0]
    while starts[-1] + WINDOW < future_frames:
        starts.append(starts[-1] + stride)
    return starts


def draw_seed(index: int, start: int, member: int, label: int) -> int:
    """Match the paper seed at start=label=0 and vary noise across windows."""
    return (
        (member + 1) * 1_000_003 + index * 9_176 + start * 104_729 + label * 170_003
    ) % (2**31 - 1)


def rollout(
    predictor: Predictor,
    sample: dict,
    output: Path,
    *,
    future_frames: int,
    stride: int,
    ensemble_size: int,
    sampling_label: int,
    save_members: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Feed back the merged physical state; average only predicted frames."""
    starts = window_starts(future_frames, stride)
    initial = np.asarray(sample["initial"], dtype=np.float32)
    sums = np.zeros((future_frames + 1, *initial.shape), dtype=np.float64)
    counts = np.zeros(future_frames + 1, dtype=np.int32)
    sums[0], counts[0] = initial, 1
    boundary = np.isin(np.asarray(sample["node_type"]).reshape(-1), (4, 6))
    windows = []
    first_window = None
    started = time.perf_counter()
    for ordinal, start in enumerate(starts):
        if counts[start] == 0:
            raise ValueError(f"no generated conditioning state at frame {start}")
        condition = (sums[start] / counts[start]).astype(np.float32)
        condition[boundary, :2] = initial[boundary, :2]
        current = {**sample, "initial": condition}
        seeds = []
        total = None
        sync(predictor.device)
        begin = time.perf_counter()
        for member in range(ensemble_size):
            seed = draw_seed(sample["trajectory_index"], start, member, sampling_label)
            seeds.append(seed)
            prediction, _ = predictor.predict(current, seed, diagnostics=False)
            if prediction.shape != (WINDOW + 1, *initial.shape):
                raise ValueError(f"unexpected prediction shape: {prediction.shape}")
            if save_members or not np.isfinite(prediction).all():
                np.savez_compressed(
                    output / f"window_{start:04d}_member_{member:02d}.npz",
                    prediction=prediction,
                    seed=seed,
                    start_frame=start,
                )
            if not np.isfinite(prediction).all():
                raise FloatingPointError(
                    f"nonfinite prediction: window {start}, member {member}"
                )
            # Preserve the native evaluator's FP32 member-mean arithmetic.
            if total is None:
                total = prediction.copy()
            else:
                total += prediction
        total /= ensemble_size
        total[0] = condition
        total[1:, boundary, :2] = initial[None, boundary, :2]
        if not np.isfinite(total).all():
            raise FloatingPointError(f"nonfinite ensemble mean at window {start}")
        sync(predictor.device)
        seconds = time.perf_counter() - begin
        stop = min(start + WINDOW, future_frames)
        # The conditioning frame is not an independent prediction and gets no vote.
        sums[start + 1 : stop + 1] += total[1 : stop - start + 1]
        counts[start + 1 : stop + 1] += 1
        if first_window is None:
            first_window = total.copy()
        np.savez_compressed(
            output / f"window_{start:04d}.npz",
            prediction=total,
            condition=condition,
            seeds=np.asarray(seeds, dtype=np.int64),
            global_frame=np.arange(start, start + WINDOW + 1),
            time=np.arange(start, start + WINDOW + 1) * DT,
        )
        row = {
            "window": ordinal,
            "start_frame": start,
            "predicted_frames": [start + 1, start + WINDOW],
            "retained_end_frame": stop,
            "condition_contributions": int(counts[start]),
            "member_seeds": seeds,
            "inference_seconds": seconds,
        }
        windows.append(row)
        write_json(output / "windows.json", windows)
        print(
            json.dumps(
                {
                    "trajectory_index": sample["trajectory_index"],
                    **row,
                    "completed_windows": len(windows),
                    "total_windows": len(starts),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    if np.any(counts == 0):
        raise ValueError("the generated trajectory contains an uncovered frame")
    merged = (sums / counts[:, None, None]).astype(np.float32)
    merged[0] = initial
    merged[1:, boundary, :2] = initial[None, boundary, :2]
    if not np.isfinite(merged).all():
        raise FloatingPointError("nonfinite merged trajectory")
    return merged, first_window, counts, windows


def render_case(
    prediction: np.ndarray,
    reference: np.ndarray,
    sample: dict,
    output: Path,
    *,
    stride: int,
    frame_stride: int,
    fps: int,
) -> None:
    """Use fixed reference scales and blank reference panels after frame 64."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.animation import FuncAnimation, PillowWriter

    points, cells = sample["points"], sample["cells"]
    weights = node_area_weights(points, cells)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)

    def quantities(fields: np.ndarray) -> list[np.ndarray]:
        pressure = fields[:, :, 2].astype(np.float64)
        pressure -= np.sum(pressure * weights, axis=1, keepdims=True) / weights.sum()
        vorticity, _, _ = triangle_vorticity_divergence(fields[:, :, :2], points, cells)
        speed = np.linalg.norm(fields[:, :, :2].astype(np.float64), axis=-1)
        return [speed, pressure, vorticity]

    predicted = quantities(prediction)
    target = quantities(reference)
    limits = []
    scale_records = []
    for name, values, truth in zip(
        ("speed", "pressure", "vorticity"), predicted, target
    ):
        bound = max(float(np.max(np.abs(truth))), 1e-12)
        low, high = (0.0, bound) if name == "speed" else (-bound, bound)
        limits.append((low, high))
        scale_records.append(
            {
                "field": name,
                "minimum": low,
                "maximum": high,
                "predicted_fraction_outside_scale": float(
                    np.mean((values < low) | (values > high))
                ),
            }
        )
    write_json(
        output / "visual_scales.json",
        {
            "source": "all retained reference frames, fixed over the full movie",
            "pressure_gauge": "subtract area-weighted spatial mean per frame",
            "scales": scale_records,
        },
    )
    figure, axes = plt.subplots(3, 2, figsize=(10, 7), layout="constrained")
    artists = []
    messages = []
    for row, (name, cmap) in enumerate(
        zip(
            ("Speed", "Pressure (zero spatial mean)", "Vorticity"),
            ("viridis", "RdBu_r", "RdBu_r"),
        )
    ):
        pair = []
        for column in range(2):
            axis = axes[row, column]
            options = {"facecolors": target[row][0]} if row == 2 else {}
            artist = axis.tripcolor(
                triangulation,
                *([] if row == 2 else [target[row][0]]),
                shading="flat" if row == 2 else "gouraud",
                cmap=cmap,
                vmin=limits[row][0],
                vmax=limits[row][1],
                **options,
            )
            axis.set_aspect("equal")
            axis.set_title(
                f"{'Reference' if column == 0 else 'Sliding window'}: {name}"
            )
            axis.set(xlabel="x", ylabel="y")
            pair.append(artist)
        messages.append(
            axes[row, 0].text(
                0.5,
                0.5,
                "No reference beyond frame 64",
                transform=axes[row, 0].transAxes,
                ha="center",
                va="center",
                visible=False,
            )
        )
        figure.colorbar(pair[1], ax=list(axes[row]), shrink=0.75, extend="both")
        artists.append(pair)
    title = figure.suptitle("")

    def update(frame: int) -> list:
        have_reference = frame < len(reference)
        for row in range(3):
            artists[row][0].set_visible(have_reference)
            messages[row].set_visible(not have_reference)
            if have_reference:
                artists[row][0].set_array(target[row][frame])
            artists[row][1].set_array(predicted[row][frame])
        title.set_text(
            f"CylinderFlow {sample['trajectory_index']} | frame {frame} | "
            f"t = {frame * DT:.2f} s | 64-frame windows / stride {stride}"
        )
        return [title]

    frames = sorted(
        set(range(0, len(prediction), frame_stride)) | {64, len(prediction) - 1}
    )
    animation = FuncAnimation(figure, update, frames=frames, blit=False)
    animation.save(output / "rollout.gif", writer=PillowWriter(fps=fps), dpi=90)
    for frame in sorted({32, 64, min(128, len(prediction) - 1), len(prediction) - 1}):
        update(frame)
        figure.savefig(output / f"frame_{frame:04d}.png", dpi=150)
    plt.close(figure)


def source_snapshot(output: Path) -> dict:
    """Keep the local inference implementation with the generated results."""
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for package in ("graph_dit", "dgn4cfd", "mgn_data"):
            for source_file in sorted((root / package).rglob("*.py")):
                archive.write(source_file, source_file.relative_to(root))
        for name in ("pyproject.toml", "SLIDING_WINDOW.md"):
            archive.write(root / name, name)
    return {"root": str(root), "base_commit": revision.stdout.strip() or None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="read the existing selection.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weights", choices=("raw", "ema_0.999", "ema_0.9999"))
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("validation", "test"), default="validation")
    parser.add_argument("--test-data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--future-frames", type=int, default=256)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--sampling-steps", type=int, default=6)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--sampling-label", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--save-members", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if args.run and (args.checkpoint is not None or args.weights is not None):
        parser.error("use either --run or the explicit --checkpoint/--weights pair")
    selected = None
    if args.run:
        selected = json.loads((args.run / "selection.json").read_text(encoding="utf-8"))
        args.checkpoint = args.run / selected["checkpoint"]
        args.weights = selected["weights"]
    if args.checkpoint is None or args.weights is None:
        parser.error("provide --run, or both --checkpoint and --weights")
    if (args.scope == "test") != (args.test_data_dir is not None):
        parser.error("--scope test requires --test-data-dir; validation must omit it")
    if args.future_frames < WINDOW or not 1 <= args.stride <= WINDOW:
        parser.error("future-frames must be >=64 and stride must be between 1 and 64")
    if (
        min(
            args.ensemble_size,
            args.case_count,
            args.threads,
            args.frame_stride,
            args.fps,
        )
        < 1
    ):
        parser.error("counts, threads, frame-stride and fps must be positive")
    if args.sampling_label < 0 or args.sampling_steps < 2:
        parser.error("sampling-label must be nonnegative and sampling-steps >=2")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("this inference entrypoint requires the assigned CUDA environment")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "request.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    try:
        run(args, device, selected)
    except Exception as error:
        write_json(
            args.output_dir / "failure.json",
            {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def run(args: argparse.Namespace, device: torch.device, selected: dict | None) -> None:
    """Load the fixed model once and retain success and failure for every case."""
    configure(device, args.threads)
    model, checkpoint = load_selected(
        args.checkpoint, args.artifacts, args.weights, inference_only=True
    )
    if model.future_frames != WINDOW:
        raise ValueError("sliding continuation requires the original 64-frame model")
    model.to(device).eval().float()
    predictor = Predictor(
        model,
        args.artifacts,
        args.data_dir,
        device,
        precision="fp32",
        sampling_steps=args.sampling_steps,
        inference_only=True,
    )
    training_identity = predictor.data.identity()
    if args.scope == "test":
        predictor.use_test_data(args.test_data_dir)
    population = list(predictor.data.splits[args.scope])
    indices = (
        args.indices if args.indices is not None else population[: args.case_count]
    )
    if (
        not indices
        or len(set(indices)) != len(indices)
        or not set(indices) <= set(population)
    ):
        raise ValueError("indices must be unique members of the declared split")
    identity = {
        "protocol": "cylinderflow.joint64.sliding.v1",
        "source": source_snapshot(args.output_dir),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_id": checkpoint.get("checkpoint_id"),
        "update": checkpoint.get("update"),
        "weights": args.weights,
        "selection": selected,
        "artifacts": str(args.artifacts.resolve()),
        "artifact_id": predictor.identity["artifact_id"],
        "representation_id": predictor.identity["representation_id"],
        "normalization": predictor.identity["normalization"],
        "training_data": training_identity,
        "evaluation_data": predictor.data.identity(),
        "scope": args.scope,
        "indices": indices,
        "population_count": len(population),
        "case_selection": "explicit indices"
        if args.indices
        else "first cases in split order",
        "observed_frames": [0],
        "reference_frames": [0, 64],
        "future_frames": args.future_frames,
        "dt": DT,
        "window_future_frames": WINDOW,
        "stride": args.stride,
        "window_starts": window_starts(args.future_frames, args.stride),
        "sampling_steps": args.sampling_steps,
        "ensemble_size": args.ensemble_size,
        "sampling_label": args.sampling_label,
        "seed_rule": "((member+1)*1000003 + index*9176 + start*104729 + label*170003) mod (2**31-1)",
        "conditioning": "merged physical UVP at the window start; re-encoded by the frozen VGAE",
        "aggregation": "physical member mean within each window, equal-weight mean across overlapping predictions",
        "boundary": "labels 4/6 retain original frame-zero UV; pressure is predicted",
        "metric_scope": "reference-backed frames 1..64 only; later frames have no error score",
        "runtime": runtime_identity(device),
    }
    write_json(args.output_dir / "identity.json", identity)
    del checkpoint
    rows = []
    started = time.perf_counter()
    for index in indices:
        output = args.output_dir / f"trajectory_{index:04d}"
        output.mkdir()
        row = {"trajectory_index": index, "prediction_complete": False, "error": None}
        begin = time.perf_counter()
        try:
            sample = predictor.load_case(index)
            prediction, first, counts, windows = rollout(
                predictor,
                sample,
                output,
                future_frames=args.future_frames,
                stride=args.stride,
                ensemble_size=args.ensemble_size,
                sampling_label=args.sampling_label,
                save_members=args.save_members,
            )
            # Future reference is first opened after all autoregressive windows finish.
            reference = predictor.data.evaluation(index)["field"]
            np.savez_compressed(
                output / "rollout.npz",
                prediction=prediction,
                target=reference,
                points=sample["points"],
                cells=sample["cells"],
                node_type=sample["node_type"],
                contribution_count=counts,
                time=np.arange(len(prediction)) * DT,
                reference_time=np.arange(len(reference)) * DT,
            )
            row["prediction_complete"] = True
            weights = node_area_weights(sample["points"], sample["cells"])
            step_rms = np.sqrt(
                np.sum(
                    weights[None, :, None]
                    * np.diff(prediction[:, :, :2].astype(np.float64), axis=0) ** 2,
                    axis=(1, 2),
                )
                / (2 * weights.sum())
            )
            row.update(
                {
                    "prediction_complete": True,
                    "inference_seconds": sum(
                        value["inference_seconds"] for value in windows
                    ),
                    "prefix64_single_window": compute_metrics(
                        first,
                        reference,
                        sample["points"],
                        sample["cells"],
                        sample["node_type"],
                        DT,
                    ),
                    "prefix64_sliding": compute_metrics(
                        prediction[:65],
                        reference,
                        sample["points"],
                        sample["cells"],
                        sample["node_type"],
                        DT,
                    ),
                    "maximum_speed": float(
                        np.linalg.norm(
                            prediction[:, :, :2].astype(np.float64), axis=-1
                        ).max()
                    ),
                    "uv_step_rms": step_rms.tolist(),
                    "render_complete": False,
                }
            )
            write_json(output / "summary.json", row)
            if not args.no_render:
                render_case(
                    prediction,
                    reference,
                    sample,
                    output,
                    stride=args.stride,
                    frame_stride=args.frame_stride,
                    fps=args.fps,
                )
                row["render_complete"] = True
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            write_json(
                output / "failure.json",
                {
                    "error": row["error"],
                    "traceback": traceback.format_exc(),
                },
            )
        row["seconds"] = time.perf_counter() - begin
        write_json(output / "summary.json", row)
        rows.append(row)
        predictor.static_graphs.clear()
        predictor.static_hops.clear()
        summary = {
            "requested_indices": indices,
            "completed_cases": len(rows),
            "prediction_complete": sum(item["prediction_complete"] for item in rows),
            "failed_cases": sum(item["error"] is not None for item in rows),
            "elapsed_seconds": time.perf_counter() - started,
            "cases": rows,
        }
        write_json(args.output_dir / "summary.json", summary)
        print(
            json.dumps(
                {
                    "trajectory_index": index,
                    "completed_cases": len(rows),
                    "requested_cases": len(indices),
                    "error": row["error"],
                }
            ),
            flush=True,
        )
    if any(row["error"] is not None for row in rows):
        raise RuntimeError(
            "one or more cases failed; retained per-case failure records"
        )


if __name__ == "__main__":
    main()
