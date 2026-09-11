"""Bounded acceptance for premature-stop and interrupted-evaluation failure modes."""

import argparse
import json
import math
from pathlib import Path
import random
import shutil
from unittest.mock import patch

import numpy as np
import torch

from .config import learning_rate, load_config
from .evaluate import Predictor, evaluate_model, load_selected
from .physical_monitor import PhysicalMonitor
from .runtime import ROOT, load_checkpoint, write_json
from .smoke import assert_equal
from .train import train_run


def acceptance(smoke_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT / "configs/h1_w512_cosine_ema_4090_20260908a.json")
    assert learning_rate(config, 4000) == 1e-4
    assert learning_rate(config, 1000000) == 1e-7
    assert learning_rate(config, 4001) < learning_rate(config, 4000)
    validation = config["validation"]

    def round_at(monitor, update, scores, failures=None):
        for name, score in zip(validation["weights"], scores):
            monitor.store_candidate(
                {
                    "update": update,
                    "weights": name,
                    "score": score,
                    "failed_clips": (failures or {}).get(name, 0),
                    "clip_count": 72,
                    "trajectory_count": 24,
                    "checkpoint_id": str(update),
                    "checkpoint": f"{update}.pt",
                }
            )
        monitor.complete(update, str(update), 72, 24)

    monitor = PhysicalMonitor(output / "strict", validation, 0)
    round_at(monitor, 5000, [0.4, 0.3, 0.35])
    assert monitor.state["best"]["weights"] == "ema_0.999"
    round_at(monitor, 10000, [0.3, 0.4, 0.4])
    assert (
        monitor.state["bad_evaluations"] == 1
        and monitor.state["best"]["update"] == 5000
    )
    smaller = math.nextafter(0.3, 0.0)
    round_at(monitor, 15000, [smaller, 0.4, 0.4])
    assert (
        monitor.state["best"]["score"] == smaller
        and monitor.state["bad_evaluations"] == 0
    )
    round_at(monitor, 20000, [0.01, None, None], {"raw": 1})
    assert (
        monitor.state["bad_evaluations"] == 1
        and monitor.state["best"]["update"] == 15000
    )
    round_at(monitor, 25000, [None, None, None])
    assert monitor.state["bad_evaluations"] == 2
    monitor.error(30000, 1, RuntimeError("evaluation interruption"))
    assert monitor.state["bad_evaluations"] == 0
    round_at(monitor, 30000, [0.4, 0.4, 0.4])
    assert monitor.state["bad_evaluations"] == 1
    replay = PhysicalMonitor(output / "strict", validation, 15000)
    assert (
        replay.state["best"]["update"] == 15000 and replay.state["bad_evaluations"] == 0
    )
    assert all(row["update"] <= 15000 for row in replay.candidates(30000))
    assert list((replay.directory / "recovery_archives").glob("*/identity.json"))

    boundary = PhysicalMonitor(output / "boundary", validation, 0)
    round_at(boundary, 395000, [0.3, 0.4, 0.4])
    for update in range(400000, 500000, 5000):
        round_at(boundary, update, [0.3, 0.4, 0.4])
    assert (
        boundary.state["bad_evaluations"] == 20 and not boundary.state["stop_requested"]
    )
    round_at(boundary, 500000, [0.3, 0.4, 0.4])
    assert boundary.state["stop_requested"]
    exact = PhysicalMonitor(output / "exact_twenty", validation, 0)
    round_at(exact, 400000, [0.3, 0.4, 0.4])
    for update in range(405000, 505000, 5000):
        round_at(exact, update, [0.3, 0.4, 0.4])
        assert exact.state["stop_requested"] == (update == 500000)

    small = json.loads((smoke_root / "continuous/config.json").read_text())
    small["training"].update(schedule_total_updates=8, schedule="cosine")
    small["validation"].update(
        weights=validation["weights"],
        early_stopping={"enabled": True, "min_updates": 6, "patience_evaluations": 2},
    )
    calls = []
    interrupted = False

    def evaluator(noisy, crash):
        def evaluate(predictor, indices, seeds, destination, provenance, **kwargs):
            nonlocal interrupted
            key = (provenance["update"], provenance["weights"])
            calls.append(key)
            if noisy:
                random.random()
                np.random.random(11)
                torch.rand(13)
                predictor.model.eval()
            if crash and key == (4, "ema_0.999") and not interrupted:
                interrupted = True
                raise RuntimeError("injected interruption after raw receipt")
            return {
                "failed_clips": 0,
                "clip_count": len(indices) * len(seeds),
                "trajectory_count": len(indices),
                "selection_uv_relative_rmse": {
                    "raw": 0.4,
                    "ema_0.999": 0.3,
                    "ema_0.9999": 0.35,
                }[key[1]],
            }

        return evaluate

    control, recovered = output / "control", output / "recovered"
    with patch("graph_dit.evaluate.evaluate_model", evaluator(False, False)):
        first_result = train_run(
            small,
            smoke_root / "artifacts",
            smoke_root / "data",
            control,
            8,
            "cpu",
            debug=True,
        )
    calls.clear()
    with patch("graph_dit.evaluate.evaluate_model", evaluator(True, True)):
        try:
            train_run(
                small,
                smoke_root / "artifacts",
                smoke_root / "data",
                recovered,
                8,
                "cpu",
                debug=True,
            )
            raise AssertionError("evaluation interruption was swallowed")
        except RuntimeError as failure:
            assert "injected interruption" in str(failure)
        second_result = train_run(
            small,
            smoke_root / "artifacts",
            smoke_root / "data",
            recovered,
            8,
            "cpu",
            resume=True,
            debug=True,
        )
        before = len(calls)
        train_run(
            small,
            smoke_root / "artifacts",
            smoke_root / "data",
            recovered,
            8,
            "cpu",
            resume=True,
            debug=True,
        )
        assert len(calls) == before
    assert first_result["state"] == second_result["state"] == "early_stopped"
    assert first_result["update"] == second_result["update"] == 6
    assert calls.count((4, "raw")) == 1
    first, second = [
        load_checkpoint(folder / "recovery_latest.pt")
        for folder in (control, recovered)
    ]
    for name in ("model", "ema", "optimizer", "training_generator", "sample_cursor"):
        assert_equal(first[name], second[name])

    # A stale later checkpoint must not be paired with newly replayed raw weights.
    shutil.copyfile(
        recovered / "checkpoints/update_000000002.pt", recovered / "recovery_latest.pt"
    )
    calls.clear()
    with patch("graph_dit.evaluate.evaluate_model", evaluator(True, False)):
        rollback_result = train_run(
            small,
            smoke_root / "artifacts",
            smoke_root / "data",
            recovered,
            8,
            "cpu",
            resume=True,
            debug=True,
        )
    assert (
        rollback_result["state"] == "early_stopped" and rollback_result["update"] == 6
    )
    assert calls == [(step, name) for step in (4, 6) for name in validation["weights"]]
    rolled = load_checkpoint(recovered / "recovery_latest.pt")
    for name in ("model", "ema", "optimizer", "training_generator", "sample_cursor"):
        assert_equal(first[name], rolled[name])

    model, _ = load_selected(
        control / "checkpoints/update_000000006.pt", smoke_root / "artifacts", "raw"
    )
    predictor = Predictor(
        model,
        smoke_root / "artifacts",
        smoke_root / "data",
        torch.device("cpu"),
        debug=True,
    )
    with patch.object(
        predictor, "predict", side_effect=RuntimeError("program failure")
    ):
        try:
            evaluate_model(
                predictor,
                (2,),
                [0],
                output / "runtime_failure",
                {},
                fail_on_runtime_error=True,
            )
            raise AssertionError("program failure became a numeric failure")
        except RuntimeError:
            pass
    with patch.object(
        predictor, "predict", side_effect=FloatingPointError("nonfinite prediction")
    ):
        failed = evaluate_model(
            predictor,
            (2,),
            [0],
            output / "numeric_failure",
            {},
            fail_on_runtime_error=True,
        )
        assert (
            failed["failed_clips"] == 1 and failed["selection_uv_relative_rmse"] is None
        )
    result = {
        "strict_decrease_and_ties": True,
        "raw_ema_single_round": True,
        "invalid_candidate_exclusion": True,
        "twenty_rounds_and_500k_boundary": True,
        "error_breaks_streak": True,
        "restore_step_filters_future_records": True,
        "rollback_archives_future_checkpoints_and_recomputes": True,
        "partial_evaluation_resume_without_duplicates": True,
        "terminal_resume_no_extra_updates": True,
        "training_raw_ema_adam_and_rng_exact": True,
        "runtime_and_numeric_failures_distinct": True,
        "cosine_warmup_and_1m_floor": True,
        "synthetic_only": True,
    }
    write_json(output / "acceptance.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(acceptance(args.smoke_root, args.output_dir), indent=2))
