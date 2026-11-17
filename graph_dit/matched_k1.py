"""Fixed-250k evaluation: three independent S6/K1 scores per trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from graph_dit.metrics import compute_metrics
from graph_dit.performance import configure, seed_draw
from graph_dit.runtime import load_checkpoint, write_json

PROTOCOL = "matched_250k_s6_k1_scores.v1"
MODELS = ("gladit", "regression", "gnn")


def mean_metrics(rows: list[dict]) -> dict:
    """Average complete scores, including per-frame numeric arrays."""
    result = {}
    for key in rows[0]:
        if key == "finite":
            continue
        values = [row[key] for row in rows]
        if isinstance(values[0], (str, dict, bool)):
            continue
        if any(value is None for value in values):
            result[key] = None
            continue
        arrays = [np.asarray(value, dtype=np.float64) for value in values]
        if any(a.shape != arrays[0].shape for a in arrays):
            raise ValueError(f"inconsistent metric shape: {key}")
        result[key] = (
            np.mean(arrays, axis=0).tolist()
            if all(np.isfinite(a).all() for a in arrays)
            else None
        )
    return result


def load_predictor(args: argparse.Namespace, state: dict, device: torch.device):
    """Reuse the selected branch's native codec, boundary rules and predictor."""
    if args.model == "gladit":
        from graph_dit.evaluate import Predictor, load_selected

        if args.artifacts is None:
            raise ValueError("GLaDiT requires --artifacts")
        model, _ = load_selected(args.checkpoint, args.artifacts, args.weights)
        model.to(device).eval()
        predictor = Predictor(
            model, args.artifacts, args.data_dir, device, "fp32", sampling_steps=6
        )
        manifest_format = predictor.data.manifest["format"]
        actual_dataset = "airfoil" if "airfoil" in manifest_format else "cylinderflow"
        if actual_dataset != args.dataset:
            raise ValueError("dataset differs from this branch's prepared data")
        if args.split == "test":
            if args.test_data_dir is None:
                raise ValueError("GLaDiT Test requires --test-data-dir")
            predictor.use_test_data(args.test_data_dir)
        return (
            predictor.data.splits[args.split],
            predictor.load_case,
            lambda sample, seed: predictor.predict(sample, seed)[0],
            lambda index: predictor.data.evaluation(index)["field"],
            predictor.data.identity(),
        )

    from deterministic_joint.data import EvaluationData

    if args.cache_dir is None:
        raise ValueError("component controls require --cache-dir")
    data = EvaluationData(
        args.dataset, args.split, args.data_dir, args.cache_dir, device
    )
    for key in ("dataset", "artifact_id", "representation_id"):
        expected = args.dataset if key == "dataset" else data.identity[key]
        if state.get(key) != expected:
            raise ValueError(f"checkpoint/cache mismatch: {key}")
    if args.model == "regression":
        from deterministic_joint.evaluation import CHECKPOINT_FORMAT, predict
        from deterministic_joint.model import JointGraphTransformer

        model = JointGraphTransformer().to(device)
        predict_case = lambda sample, seed: predict(model, data, sample)
    else:
        from graph_latent_diffusion import CHECKPOINT_FORMAT
        from graph_latent_diffusion.model import GraphLatentDiffusion

        model = GraphLatentDiffusion().to(device)

        def predict_case(sample, seed):
            z0, context, graph, decoder = data.encode(sample)
            generator = torch.Generator(device=device).manual_seed(seed)
            future = model.sample(
                z0, context, graph, sampling_steps=6, generator=generator
            )
            return data.decode(future, sample, decoder)

    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("wrong checkpoint model format")
    if any(
        state["config"]["model"].get(k) != v for k, v in model.architecture().items()
    ):
        raise ValueError("checkpoint architecture mismatch")
    model.load_state_dict(state["ema"][args.weights], strict=True)
    model.eval()
    return data.indices, data.initial, predict_case, data.truth, data.source_identity


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    """Retain every requested sample and fail the report on any incomplete case."""
    args.checkpoint = (args.run / args.checkpoint).resolve()
    if not args.checkpoint.is_relative_to(args.run.resolve()):
        raise ValueError("checkpoint must belong to --run")
    state = load_checkpoint(args.checkpoint)
    run_id = state.get("run_id") or str(state.get("checkpoint_id", "")).split(":")[0]
    if run_id != args.run_id or state.get("update") != 250000:
        raise ValueError("expected the specified run's fixed 250000-update checkpoint")
    if state["config"].get("seed") != 0:
        raise ValueError("the matched ablation uses training seed 0")
    if state.get("format") == "graph_dit.inference.v1":
        if state.get("weights") != args.weights:
            raise ValueError("inference export has different EMA weights")
    elif args.weights not in state.get("ema", {}):
        raise ValueError("checkpoint is missing EMA 0.999")
    device = torch.device("cuda:0")
    configure(device)
    indices, initial, predict_case, truth, source = load_predictor(args, state, device)
    indices = [int(index) for index in indices]
    if len(indices) != 100 or len(set(indices)) != 100:
        raise ValueError("expected all 100 distinct official trajectories")
    labels = [0] if args.model == "regression" else [0, 1, 2]
    identity = {
        "protocol": PROTOCOL,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "update": 250000,
        "weights": args.weights,
        "run_id": run_id,
        "checkpoint": str(args.checkpoint),
        "training_seed": 0,
        "representation_id": state.get("representation_id"),
        "artifact_id": state["artifact_id"],
        "source": source,
        "future_frames": 64,
        "indices": indices,
        "sampling_labels": labels,
        "sampling_steps": 0 if args.model == "regression" else 6,
        "ensemble_size": 1,
        "independent_repeats": len(labels),
        "deterministic": args.model == "regression",
        "seed_rule": "((label+1)*1000003 + trajectory_index*9176) % (2**31-1)",
        "aggregation": "score each K1; average repeats within trajectory; equal trajectory mean",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "identity.json", identity)
    trajectories, samples = [], []
    for index in indices:
        members = []
        for label in labels:
            seed = None if args.model == "regression" else seed_draw(index, label)
            row = {
                "trajectory_index": index,
                "sampling_label": label,
                "prng_seed": seed,
                "finite": False,
            }
            try:
                sample = initial(index)
                if int(sample["trajectory_index"]) != index:
                    raise ValueError("trajectory registry mismatch")
                field = predict_case(sample, seed)
                if field.shape != (65, len(sample["points"]), 3):
                    raise ValueError(f"invalid field shape: {field.shape}")
                if not np.array_equal(field[0], sample["initial"]):
                    raise ValueError("initial state changed")
                if not np.isfinite(field).all():
                    raise FloatingPointError("nonfinite prediction")
                target = truth(index)
                metrics = compute_metrics(
                    field,
                    target,
                    sample["points"],
                    sample["cells"],
                    sample["node_type"],
                    0.08 if args.dataset == "cylinderflow" else 0.0016,
                )
                if args.dataset == "cylinderflow":
                    metrics.pop("airfoil_surface_uv_rmse", None)
                    wall = np.asarray(sample["node_type"]).reshape(-1) == 6
                    if wall.any():
                        delta = (
                            field[1:, wall, :2].astype(np.float64)
                            - target[1:, wall, :2]
                        )
                        metrics["wall_uv_rmse"] = float(np.sqrt(np.mean(delta**2)))
                row["metrics"] = metrics
                row["finite"] = bool(
                    metrics.get("finite") and np.isfinite(metrics["uv_relative_rmse"])
                )
                del field, target
            except Exception as error:
                row["failure"] = f"{type(error).__name__}: {error}"
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            write_json(args.output_dir / f"case_{index:05d}_sample{label}.json", row)
            members.append(row)
            samples.append(row)
        finite = all(row["finite"] for row in members)
        trajectory = {
            "trajectory_index": index,
            "finite": finite,
            "sample_count": len(members),
            "metrics": mean_metrics([row["metrics"] for row in members])
            if finite
            else None,
        }
        write_json(args.output_dir / f"trajectory_{index:05d}.json", trajectory)
        trajectories.append(trajectory)
    failures = [row for row in samples if not row["finite"]]
    summary = {
        **identity,
        "complete": not failures,
        "trajectory_count": len(trajectories),
        "sample_count": len(samples),
        "failure_count": len(failures),
        "successful_sample_count": len(samples) - len(failures),
        "failed_trajectories": sum(not row["finite"] for row in trajectories),
        "metrics": mean_metrics([row["metrics"] for row in trajectories])
        if not failures
        else None,
    }
    write_json(args.output_dir / "failures.json", {"failures": failures})
    write_json(args.output_dir / "summary.json", summary)
    if failures:
        raise SystemExit(
            "Incomplete population; inspect saved failures and individual scores"
        )


def collect(args: argparse.Namespace) -> None:
    """Accept only a complete matched table, without mixing K8 or other weights."""
    rows, populations, runs = {}, {}, {}
    for filename in args.summaries:
        result = json.loads(filename.read_text(encoding="utf-8"))
        model, dataset, split = (result[k] for k in ("model", "dataset", "split"))
        key = (dataset, split, model)
        repeats = 1 if model == "regression" else 3
        expected = {
            "protocol": PROTOCOL,
            "update": 250000,
            "weights": "ema_0.999",
            "future_frames": 64,
            "ensemble_size": 1,
            "independent_repeats": repeats,
            "sampling_labels": [0] if repeats == 1 else [0, 1, 2],
            "sampling_steps": 0 if repeats == 1 else 6,
            "complete": True,
            "failure_count": 0,
            "trajectory_count": 100,
            "training_seed": 0,
            "sample_count": 100 * repeats,
        }
        if key in rows or any(result.get(k) != v for k, v in expected.items()):
            raise ValueError(
                f"duplicate, incomplete or incompatible summary: {filename}"
            )
        population = (
            result["indices"],
            result["artifact_id"],
            result["representation_id"],
        )
        previous = populations.setdefault((dataset, split), population)
        if population != previous or not all(population[1:]):
            raise ValueError(
                "population or frozen representation differs between models"
            )
        if runs.setdefault((dataset, model), result["run_id"]) != result["run_id"]:
            raise ValueError("Validation and Test must use the same training run")
        rows[key] = {
            "dataset": dataset,
            "split": split,
            "model": model,
            "uv_relative_rmse_percent": 100 * result["metrics"]["uv_relative_rmse"],
            "repeats": repeats,
            "summary": str(filename),
        }
    required = {
        (d, s, m)
        for d in ("cylinderflow", "airfoil")
        for s in ("validation", "test")
        for m in MODELS
    }
    if set(rows) != required:
        raise ValueError(
            f"expected exactly 12 entries; missing {sorted(required - set(rows))}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ordered = [rows[key] for key in sorted(rows)]
    with (args.output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    lines = [
        "# Matched 250K / EMA 0.999",
        "",
        "Independent K1 scores averaged over three repetitions, then equally over trajectories. "
        "Diffusion: S6. Regression: one deterministic prediction. No physical-field averaging.",
        "",
        "| Dataset | Split | Model | UV error (%) |",
        "| --- | --- | --- | ---: |",
    ]
    lines.extend(
        f"| {r['dataset']} | {r['split']} | {r['model']} | {r['uv_relative_rmse_percent']:.4f} |"
        for r in ordered
    )
    (args.output_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--model", choices=MODELS, required=True)
    run.add_argument("--dataset", choices=("cylinderflow", "airfoil"), required=True)
    run.add_argument("--split", choices=("validation", "test"), required=True)
    for name in ("run", "checkpoint", "data-dir", "output-dir"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--weights", choices=("ema_0.999",), default="ema_0.999")
    for name in ("artifacts", "cache-dir", "test-data-dir"):
        run.add_argument("--" + name, type=Path)
    summary = sub.add_parser("collect")
    summary.add_argument("--summaries", nargs="+", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
