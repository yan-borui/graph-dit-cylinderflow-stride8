"""Standalone shared-scale mesh movies and structured-log training curves."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Rectangle
import numpy as np

from .metrics import node_area_weights, triangle_vorticity_divergence


def _write_json(destination: Path, record: dict[str, Any]) -> None:
    destination.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _channels(field, points, cells, weights):
    pressure = field[..., 2].astype(float)
    pressure -= (pressure @ weights / weights.sum())[:, None]
    vorticity, _, _ = triangle_vorticity_divergence(field[..., :2], points, cells)
    return [np.linalg.norm(field[..., :2], axis=-1), pressure, vorticity]


def render(
    inputs: list[Path],
    output_dir: Path,
    labels: list[str] | None = None,
    every: int = 1,
    fps: float = 12.5,
    scales_file: Path | None = None,
    snapshots: list[int] | None = None,
    snapshot_pdf: bool = False,
    *,
    title: str = "",
    paired_rows: bool = False,
    frames: list[int] | None = None,
    viewport: list[float] | None = None,
) -> dict[str, Any]:
    """Render common-scale mesh fields and errors, with optional compact columns."""
    if every < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("every and fps must be positive")
    if frames is not None and (
        not frames
        or len(set(frames)) != len(frames)
        or any(frame < 0 or frame > 64 for frame in frames)
    ):
        raise ValueError("frames must be distinct stored indices in 0..64")
    if snapshots is not None and any(frame < 0 or frame > 64 for frame in snapshots):
        raise ValueError("snapshot frames must be in 0..64")
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    bundles = []
    for file_name in inputs:
        with np.load(file_name, allow_pickle=False) as handle:
            bundles.append({key: handle[key] for key in handle.files})
    first = bundles[0]
    points, cells, target = first["points"], first["cells"], first["target"]
    visible_nodes = np.ones(len(points), dtype=bool)
    visible_cells = np.ones(len(cells), dtype=bool)
    if viewport is not None:
        bounds = np.asarray(viewport, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError("viewport must contain four finite bounds")
        xmin, xmax, ymin, ymax = bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("viewport bounds must be increasing")
        vertices = points[cells]
        visible_cells = (
            (vertices[..., 0].max(axis=1) >= xmin)
            & (vertices[..., 0].min(axis=1) <= xmax)
            & (vertices[..., 1].max(axis=1) >= ymin)
            & (vertices[..., 1].min(axis=1) <= ymax)
        )
        if not visible_cells.any():
            raise ValueError("viewport does not intersect the mesh")
        visible_nodes[:] = False
        visible_nodes[np.unique(cells[visible_cells])] = True
    for bundle in bundles:
        for key in ("points", "cells", "target", "physical_time", "raw_frame_indices"):
            if not np.array_equal(bundle[key], first[key]):
                raise ValueError(f"paired rendering requires identical {key}")
        if bundle["prediction"].shape != target.shape or target.shape[0] != 65:
            raise ValueError("render expects complete [65,N,3] predictions")
        if not np.isfinite(bundle["prediction"]).all():
            raise ValueError(
                "nonfinite prediction: retain its failure record; render a finite case"
            )
    labels = labels or [f"prediction {number + 1}" for number in range(len(bundles))]
    if len(labels) != len(bundles):
        raise ValueError("provide one label per input")
    weights = node_area_weights(points, cells)
    truth = _channels(target, points, cells, weights)
    predictions = [
        _channels(bundle["prediction"], points, cells, weights) for bundle in bundles
    ]
    errors = []
    for bundle, fields in zip(bundles, predictions):
        errors.append(
            [
                np.linalg.norm(
                    bundle["prediction"][..., :2] - target[..., :2], axis=-1
                ),
                np.abs(fields[1] - truth[1]),
                np.abs(fields[2] - truth[2]),
            ]
        )
    names = ["speed", "gauge_free_pressure", "vorticity"]
    error_names = ["UV vector error", "gauge-free pressure error", "vorticity error"]
    scales = {}
    for row, name in enumerate(names):
        visible = visible_cells if row == 2 else visible_nodes
        maximum = max(
            float(np.max(np.abs(fields[row][:, visible])))
            for fields in [truth] + predictions
        )
        maximum = max(maximum, 1e-12)
        error_max = max(
            max(float(np.max(fields[row][:, visible])) for fields in errors), 1e-12
        )
        scales[name] = {
            "vmin": 0 if row == 0 else -maximum,
            "vmax": maximum,
            "error_max": error_max,
            "error_quantity": error_names[row],
        }
    if scales_file:
        scales = json.loads(Path(scales_file).read_text())["scales"]
    _write_json(
        output_dir / "scales.json",
        {
            "scales": scales,
            "meaning": "fixed across all 65 times and supplied methods; pressure is area-gauge-free",
            "inputs": [str(item) for item in inputs],
            "viewport": viewport,
            "spatial_scope": (
                "triangles intersecting viewport and their nodes"
                if viewport is not None
                else "full domain"
            ),
        },
    )
    triangles = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    selected = (
        sorted(frames)
        if frames is not None
        else sorted(set(range(0, 65, every)) | {64})
    )
    snapshot_frames = (
        {0, selected[len(selected) // 2], 64} if snapshots is None else set(snapshots)
    )
    figure, axes = plt.subplots(
        6 if paired_rows else 3,
        1 + len(bundles) if paired_rows else 1 + 2 * len(bundles),
        figsize=(24, 14) if paired_rows else (5 * (1 + 2 * len(bundles)), 8),
        dpi=100,
        squeeze=False,
    )
    artists = []
    for row, name in enumerate(names):
        panels = [(row * 2 if paired_rows else row, 0, truth[row], "GT", False)]
        if paired_rows:
            axes[row * 2 + 1, 0].set_axis_off()
            axes[row * 2 + 1, 0].text(
                0.5,
                0.5,
                f"{error_names[row]}\nShared scale across methods",
                ha="center",
                va="center",
                transform=axes[row * 2 + 1, 0].transAxes,
            )
        for column, (label, pred, err) in enumerate(
            zip(labels, predictions, errors), 1
        ):
            panels.extend(
                [
                    (row * 2, column, pred[row], label, False),
                    (row * 2 + 1, column, err[row], f"{label}: error", True),
                ]
                if paired_rows
                else [
                    (row, column * 2 - 1, pred[row], label, False),
                    (row, column * 2, err[row], f"{label}: error", True),
                ]
            )
        for plot_row, column, values, label, is_error in panels:
            ax = axes[plot_row, column]
            limits = scales[name]
            color_values = () if row == 2 else (values[0],)
            artist = ax.tripcolor(
                triangles,
                *color_values,
                **({"facecolors": values[0]} if row == 2 else {}),
                shading="flat" if row == 2 else "gouraud",
                cmap="magma" if is_error else "viridis" if row == 0 else "RdBu_r",
                vmin=0 if is_error else limits["vmin"],
                vmax=limits["error_max"] if is_error else limits["vmax"],
            )
            artists.append((artist, values))
            quantity = error_names[row] if is_error else name.replace("_", " ")
            ax.set_title(f"{label} | {quantity}", fontsize=10)
            ax.set_aspect("equal")
            if viewport is not None:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            low = 0 if is_error else limits["vmin"]
            high = limits["error_max"] if is_error else limits["vmax"]
            visible = visible_cells if row == 2 else visible_nodes
            under = bool(np.any(values[:, visible] < low))
            over = bool(np.any(values[:, visible] > high))
            extend = (
                "both"
                if under and over
                else "min"
                if under
                else "max"
                if over
                else "neither"
            )
            figure.colorbar(artist, ax=ax, shrink=0.65, extend=extend)
    if viewport is not None:
        overview = (
            axes[1, 0] if paired_rows else axes[0, 0].inset_axes([0.65, 0.65, 0.3, 0.3])
        )
        overview.clear()
        overview.set_axis_on()
        overview.set_aspect("equal")
        overview.triplot(triangles, color="0.65", linewidth=0.1)
        overview.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="red",
                linewidth=1.2,
            )
        )
        overview.set_title("Full domain | red box: displayed region", fontsize=9)
        overview.set_xlabel("x")
        overview.set_ylabel("y")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    with imageio.get_writer(
        output_dir / "comparison.gif", mode="I", duration=1000 * every / fps, loop=0
    ) as gif:
        with imageio.get_writer(
            output_dir / "comparison.mp4",
            fps=fps / every,
            codec="libx264",
            macro_block_size=2,
            quality=8,
            pixelformat="yuv420p",
            output_params=["-profile:v", "high", "-movflags", "+faststart"],
        ) as video:
            for frame in sorted(set(selected) | snapshot_frames):
                for artist, values in artists:
                    artist.set_array(values[frame])
                figure.suptitle(
                    (f"{title}\n" if title else "")
                    + f"stored frame {frame}/64 | raw index {int(first['raw_frame_indices'][frame])} | "
                    f"t={float(first['physical_time'][frame]):.4g} s",
                    fontsize=13,
                )
                figure.canvas.draw()
                pixels = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
                if frame in selected:
                    gif.append_data(pixels)
                    video.append_data(pixels)
                if frame in snapshot_frames:
                    figure.savefig(output_dir / f"frame_{frame:03d}.png", dpi=120)
                    if snapshot_pdf:
                        figure.savefig(output_dir / f"frame_{frame:03d}.pdf")
    plt.close(figure)
    result = {
        "frames": selected,
        "snapshot_frames": sorted(snapshot_frames),
        "render_seconds": time.perf_counter() - started,
        "inference_included": False,
        "title": title,
        "viewport": viewport,
        "fps": fps / every,
        "physical_time": [float(first["physical_time"][frame]) for frame in selected],
        "reduced_frames": len(selected) != 65,
        "files": ["comparison.gif", "comparison.mp4", "scales.json"]
        + [f"frame_{frame:03d}.png" for frame in sorted(snapshot_frames)]
        + (
            [f"frame_{frame:03d}.pdf" for frame in sorted(snapshot_frames)]
            if snapshot_pdf
            else []
        ),
    }
    _write_json(output_dir / "render.json", result)
    return result


def plot_curves(runs, output_dir):
    from .runtime import read_jsonl

    output_dir.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(2, 3, figsize=(15, 7))
    for directory in runs:
        records = read_jsonl(directory / "metrics.jsonl")
        train = [row for row in records if row.get("event") == "train"]
        validation = [row for row in records if row.get("event") == "validation"]
        label = directory.name
        loss_keys = sorted(
            {
                key
                for row in train
                for key in row
                if "mse" in key or key == "kl_weighted"
            }
        )
        for key in loss_keys or ["loss"]:
            rows = [row for row in train if row.get(key) is not None]
            axes[0, 0].plot(
                [r["update"] for r in rows],
                [r[key] for r in rows],
                label=f"{label}:{key}",
            )
        for ax, rows, key in [
            (axes[0, 1], validation, "uv_relative_rmse"),
            (axes[0, 2], validation, "failed_clips"),
            (axes[1, 0], train, "learning_rate"),
            (axes[1, 1], train, "gradient_norm"),
            (axes[1, 2], train, "elapsed_seconds"),
        ]:
            rows = [row for row in rows if row.get(key) is not None]
            ax.plot(
                [r["update"] for r in rows],
                [r[key] for r in rows],
                label=label,
                marker=".",
            )
            ax.set_title(key.replace("_", " "))
        for event in read_jsonl(directory / "events.jsonl"):
            if event["event"] == "resume":
                for ax in axes.flat:
                    ax.axvline(
                        event["update"],
                        color="gray",
                        linestyle=":",
                        label=f"{label} resume",
                    )
        selector = directory / "selector.json"
        if selector.exists():
            update = json.loads(selector.read_text())["update"]
            for ax in axes.flat:
                ax.axvline(
                    update, color="green", linestyle="--", label=f"{label} selected"
                )
    axes[0, 0].set_title("training loss components")
    for ax in axes.flat:
        ax.set_xlabel("optimizer updates")
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=140)
    plt.close(figure)
    return {"file": "training_curves.png", "source_runs": [str(item) for item in runs]}
