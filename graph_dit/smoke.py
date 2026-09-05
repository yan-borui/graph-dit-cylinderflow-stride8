"""Bounded CPU acceptance of the actual train/resume/predict/report paths."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import torch

from .ae import train_ae
from .config import load_config
from .data import FORMAT
from .evaluate import Predictor, load_selected
from .metrics import compute_metrics
from .representation import prepare, paths
from .report import report_run
from .runtime import ROOT, load_checkpoint, write_json
from .score import score
from .train import EMA, train_run


def fixture(directory: Path) -> None:
    """Synthetic analytic trajectories; never accepted as formal Train/Validation."""
    directory.mkdir(parents=True, exist_ok=False)
    data_file, manifest_file = paths(directory)
    trajectories, train_fields = [], []
    with h5py.File(data_file, "x") as handle:
        handle.attrs.update(format=FORMAT, trajectory_count=3, frames=75)
        for index, nx in enumerate((12, 14, 13)):
            ny = 12
            points = np.array(
                [(x, y) for y in np.linspace(0, 1, ny) for x in np.linspace(0, 2, nx)],
                dtype=np.float32,
            )
            cells = []
            for y in range(ny - 1):
                for x in range(nx - 1):
                    a = y * nx + x
                    cells.extend([[a, a + 1, a + nx], [a + 1, a + nx + 1, a + nx]])
            cells = np.asarray(cells, dtype=np.int64)
            labels = np.zeros(len(points), dtype=np.int64)
            labels[:nx] = labels[-nx:] = 6
            labels[::nx], labels[nx - 1 :: nx] = 4, 5
            t = np.arange(75, dtype=np.float32)[:, None] * 0.08
            x, y = points[:, 0][None], points[:, 1][None]
            u = 1 + 0.15 * np.sin(t * 2) + 0.1 * y + np.zeros_like(x)
            v = 0.1 * np.cos(t * 2) - 0.1 * x + np.zeros_like(y)
            pressure = 0.2 * x - 0.1 * y + 0.1 * np.sin(t)
            values = np.stack((u, v, pressure), -1).astype(np.float32)
            boundary = np.isin(labels, [4, 6])
            values[:, boundary, :2] = values[0, boundary, :2]
            name = f"trajectory_{index:04d}"
            group = handle.create_group(name)
            group.attrs["inlet_velocity"] = 1.0
            for key, array in (
                ("uvp", values),
                ("mesh_pos", points),
                ("cells", cells),
                ("node_type", labels),
            ):
                group.create_dataset(key, data=array)
            trajectories.append({"group": name})
            if index < 2:
                train_fields.append(values.reshape(-1, 3))
    joined = np.concatenate(train_fields).astype(np.float64)
    write_json(
        manifest_file,
        {
            "format": FORMAT,
            "frames": 75,
            "temporal_stride": 8,
            "frame_dt": 0.08,
            "raw_frame_dt": 0.01,
            "test_accessed": False,
            "splits": {"train": [0, 1], "validation": [2]},
            "trajectories": trajectories,
            "train_only_normalization": {
                "field_mean": joined.mean(0).tolist(),
                "field_std": joined.std(0).tolist(),
                "inlet_mean": 1.0,
                "inlet_std": 1.0,
            },
        },
    )


def assert_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_equal(first, second)
    else:
        assert left == right


def smoke(output: Path, movies: bool) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    fixture(output / "data")
    train_ae(output / "data", output / "ae", "cpu", epochs=1, debug=True)
    prepare(
        output / "data", output / "ae/best.pt", output / "artifacts", "cpu", debug=True
    )
    config = load_config(ROOT / "configs/base.json")
    config["model"].update(width=16, blocks=1)
    config["training"].update(
        warmup_updates=0,
        learning_rate=1e-4,
        schedule_total_updates=4,
        schedule="late_decay",
        decay_start_updates=2,
        decay_period_updates=1,
        decay_factor=0.1,
        checkpoint_every_updates=2,
        recovery_every_updates=1,
        log_every_updates=1,
    )
    config["validation"].update(every_updates=2, weights=["raw"])
    continuous, resumed = output / "continuous", output / "resumed"
    train_run(
        config, output / "artifacts", output / "data", continuous, 4, "cpu", debug=True
    )
    train_run(
        config, output / "artifacts", output / "data", resumed, 2, "cpu", debug=True
    )
    train_run(
        config,
        output / "artifacts",
        output / "data",
        resumed,
        4,
        "cpu",
        resume=True,
        debug=True,
    )
    first, second = [
        load_checkpoint(directory / "recovery_latest.pt")
        for directory in (continuous, resumed)
    ]
    for key in ("model", "ema", "optimizer", "training_generator", "sample_cursor"):
        assert_equal(first[key], second[key])
    changed = deepcopy(config)
    changed["training"]["learning_rate"] /= 2
    try:
        train_run(
            changed,
            output / "artifacts",
            output / "data",
            resumed,
            4,
            "cpu",
            resume=True,
            debug=True,
        )
        raise AssertionError("changed LR was accepted on resume")
    except ValueError:
        pass
    scalar = torch.nn.Linear(1, 1, bias=False)
    scalar.weight.data.zero_()
    averaging = EMA(scalar, [0.5])
    scalar.weight.data.fill_(2)
    averaging.update(scalar)
    scalar.weight.data.fill_(4)
    averaging.update(scalar)
    torch.testing.assert_close(
        averaging.states["ema_0.5"]["weight"], torch.tensor([[2.5]])
    )
    model, _ = load_selected(
        resumed / "checkpoints/update_000000004.pt", output / "artifacts", "ema_0.999"
    )
    predictor = Predictor(
        model, output / "artifacts", output / "data", torch.device("cpu"), debug=True
    )
    sample = predictor.load_case(2)
    predicted, _ = predictor.predict(sample, 17)
    forecast_only, no_diagnostics = predictor.predict(sample, 17, diagnostics=False)
    np.testing.assert_array_equal(predicted, forecast_only)
    assert no_diagnostics is None
    assert (
        predicted.shape == (65, len(sample["points"]), 3)
        and np.isfinite(predicted).all()
    )
    assert np.array_equal(predicted[0], sample["initial"])
    fixed = np.isin(sample["node_type"], [4, 6])
    assert np.array_equal(
        predicted[1:, fixed, :2],
        np.broadcast_to(sample["initial"][fixed, :2], predicted[1:, fixed, :2].shape),
    )
    # Alter Validation futures on disk. A fresh predictor must remain unchanged.
    with h5py.File(paths(output / "data")[0], "r+") as handle:
        original = np.asarray(handle["trajectory_0002/uvp"][1:])
        handle["trajectory_0002/uvp"][1:] = original + 1234
    other = Predictor(
        model, output / "artifacts", output / "data", torch.device("cpu"), debug=True
    )
    poisoned, _ = other.predict(other.load_case(2), 17)
    np.testing.assert_array_equal(predicted, poisoned)
    with h5py.File(paths(output / "data")[0], "r+") as handle:
        handle["trajectory_0002/uvp"][1:] = original
    target = predictor.data.evaluation(2)["field"]
    zero = np.zeros_like(target)
    independent_metric = compute_metrics(
        zero, target, sample["points"], sample["cells"], sample["node_type"], 0.08
    )
    assert np.isclose(independent_metric["uv_relative_rmse"], 1.0)
    selected = json.loads((resumed / "selection.json").read_text())
    summary_file = resumed / selected["summary"]
    original_summary = json.loads(summary_file.read_text())
    rescored = score(
        sorted((summary_file.parent / "predictions").glob("*.npz")), output / "rescored"
    )
    assert (
        original_summary["selection_uv_relative_rmse"]
        == rescored["selection_uv_relative_rmse"]
    )
    report_run(resumed, output / "review", movies=movies)
    result = {
        "synthetic_only": True,
        "vgae_training_and_cache": True,
        "continuous_resume_exact_cpu": True,
        "raw_ema_optimizer_and_noise_state_equal": True,
        "changed_resume_lr_rejected": True,
        "ema_independent_arithmetic": True,
        "future_truth_isolation": True,
        "joint64_ema_prediction_and_boundary": True,
        "common_offline_rescore": True,
        "curves_and_run_card": True,
        "movies": movies,
        "formal_gpu_quality_verified": False,
    }
    result["forecast_only_matches_quality_path"] = True
    write_json(output / "acceptance.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--movies", action="store_true")
    args = parser.parse_args()
    print(json.dumps(smoke(args.output_dir, args.movies), indent=2))


if __name__ == "__main__":
    main()
