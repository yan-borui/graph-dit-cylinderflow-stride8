"""Standalone shared-scale mesh movies and structured-log training curves."""

from __future__ import annotations

import json
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from .metrics import node_area_weights, triangle_vorticity_divergence
from .runtime import read_jsonl, write_json


def _channels(field, points, cells, weights):
    pressure = field[..., 2].astype(float)
    pressure -= (pressure @ weights / weights.sum())[:, None]
    vorticity, _, _ = triangle_vorticity_divergence(field[..., :2], points, cells)
    return [np.linalg.norm(field[..., :2], axis=-1), pressure, vorticity]


def render(inputs, output_dir, labels=None, every=1, fps=12.5, scales_file=None):
    if every < 1 or fps <= 0:
        raise ValueError("every and fps must be positive")
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    bundles = []
    for file_name in inputs:
        with np.load(file_name, allow_pickle=False) as handle:
            bundles.append({key: handle[key] for key in handle.files})
    first = bundles[0]
    points, cells, target = first["points"], first["cells"], first["target"]
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
    scales = {}
    for row, name in enumerate(names):
        maximum = max(
            float(np.max(np.abs(fields[row]))) for fields in [truth] + predictions
        )
        maximum = max(maximum, 1e-12)
        error_max = max(max(float(np.max(fields[row])) for fields in errors), 1e-12)
        scales[name] = {
            "vmin": 0 if row == 0 else -maximum,
            "vmax": maximum,
            "error_max": error_max,
        }
    if scales_file:
        scales = json.loads(Path(scales_file).read_text())["scales"]
    write_json(
        output_dir / "scales.json",
        {
            "scales": scales,
            "meaning": "fixed across all 65 times and supplied methods; pressure is area-gauge-free",
            "inputs": [str(item) for item in inputs],
        },
    )
    triangles = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    selected = sorted(set(range(0, 65, every)) | {64})
    figure, axes = plt.subplots(
        3, 1 + 2 * len(bundles), figsize=(5 * (1 + 2 * len(bundles)), 8), squeeze=False
    )
    artists = []
    for row, name in enumerate(names):
        panels = [(truth[row], "GT", False)]
        for label, pred, err in zip(labels, predictions, errors):
            panels.extend(
                [(pred[row], label, False), (err[row], f"{label}: error", True)]
            )
        for column, (values, label, is_error) in enumerate(panels):
            ax = axes[row, column]
            limits = scales[name]
            artist = ax.tripcolor(
                triangles,
                values[0],
                shading="flat" if row == 2 else "gouraud",
                cmap="magma" if is_error else "viridis" if row == 0 else "RdBu_r",
                vmin=0 if is_error else limits["vmin"],
                vmax=limits["error_max"] if is_error else limits["vmax"],
            )
            artists.append((artist, values))
            ax.set_title(f"{label} | {name.replace('_', ' ')}", fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            figure.colorbar(artist, ax=ax, shrink=0.65)
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
        ) as video:
            for frame in selected:
                for artist, values in artists:
                    artist.set_array(values[frame])
                figure.suptitle(
                    f"stored frame {frame}/64 | raw index {frame * 8} | t={frame * 0.08:.2f} s",
                    fontsize=13,
                )
                figure.canvas.draw()
                pixels = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
                gif.append_data(pixels)
                video.append_data(pixels)
                if frame in {0, selected[len(selected) // 2], 64}:
                    figure.savefig(output_dir / f"frame_{frame:03d}.png", dpi=120)
    plt.close(figure)
    result = {
        "frames": selected,
        "render_seconds": time.perf_counter() - started,
        "inference_included": False,
        "files": ["comparison.gif", "comparison.mp4", "scales.json"],
    }
    write_json(output_dir / "render.json", result)
    return result


def plot_curves(runs, output_dir):
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
