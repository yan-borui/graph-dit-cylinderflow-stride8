"""Durable, replayable physical checkpoint selection and authorized early stopping."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import uuid

from .runtime import clean_json, write_json


def validate_policy(validation: dict) -> None:
    policy = validation.get("early_stopping")
    if policy is None:
        return
    if not isinstance(policy, dict) or not isinstance(policy.get("enabled"), bool):
        raise ValueError("early_stopping requires a boolean enabled setting")
    for name in ("min_updates", "patience_evaluations"):
        value = policy.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"early_stopping.{name} must be a positive integer")


class PhysicalMonitor:
    """One completed round includes every requested raw/EMA evaluation."""

    def __init__(self, run: Path, validation: dict, restored_update: int):
        self.run = run
        self.directory = run / "physical_monitor"
        self.validation = deepcopy(validation)
        self.policy = deepcopy(validation.get("early_stopping", {"enabled": False}))
        self.weights = list(validation["weights"])
        self.interval = validation["every_updates"]
        for name in ("candidates", "rounds", "errors"):
            (self.directory / name).mkdir(parents=True, exist_ok=True)
        self.archive_after(restored_update)
        self.rebuild(restored_update)

    def archive_after(self, restored_update: int) -> None:
        """Preserve receipts/checkpoints beyond the recovery cursor outside selection."""
        future_files = []
        for name in ("candidates", "rounds", "errors"):
            for item in (self.directory / name).glob("*.json"):
                if json.loads(item.read_text())["update"] > restored_update:
                    future_files.append(item)
        for item in (self.run / "checkpoints").glob("update_*.pt"):
            if int(item.stem.split("_")[-1]) > restored_update:
                future_files.append(item)
        if not future_files:
            return
        archive = self.directory / "recovery_archives" / str(uuid.uuid4())
        write_json(
            archive / "identity.json",
            {
                "restored_update": restored_update,
                "files": [str(item.relative_to(self.run)) for item in future_files],
                "reason": "artifacts_after_recovery_cursor",
            },
        )
        for item in future_files:
            destination = archive / item.relative_to(self.run)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(item, destination)

    def candidates(self, update: int) -> list[dict]:
        records = [
            json.loads(item.read_text())
            for item in sorted((self.directory / "candidates").glob("*.json"))
        ]
        return [row for row in records if row["update"] <= update]

    def store_candidate(self, row: dict) -> None:
        file_name = f"update_{row['update']:09d}_{row['weights']}.json"
        write_json(self.directory / "candidates" / file_name, row)
        self.export_candidates(row["update"])

    def export_candidates(self, update: int) -> None:
        temporary = self.run / "candidates.jsonl.tmp"
        temporary.write_text(
            "".join(
                json.dumps(clean_json(row), sort_keys=True) + "\n"
                for row in self.candidates(update)
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.run / "candidates.jsonl")

    def complete(
        self,
        update: int,
        checkpoint_id: str,
        expected_clips: int,
        expected_trajectories: int,
    ) -> None:
        rows = [
            row
            for row in self.candidates(update)
            if row["update"] == update and row["checkpoint_id"] == checkpoint_id
        ]
        if len(rows) != len(self.weights) or {row["weights"] for row in rows} != set(
            self.weights
        ):
            raise RuntimeError("cannot count an incomplete raw/EMA physical round")
        eligible = [
            row
            for row in rows
            if row["failed_clips"] == 0
            and row.get("clip_count") == expected_clips
            and row.get("trajectory_count") == expected_trajectories
            and row["score"] is not None
            and math.isfinite(row["score"])
        ]
        best = (
            min(
                eligible,
                key=lambda row: (row["score"], self.weights.index(row["weights"])),
            )
            if eligible
            else None
        )
        write_json(
            self.directory / "rounds" / f"update_{update:09d}.json",
            {
                "update": update,
                "checkpoint_id": checkpoint_id,
                "status": "complete",
                "best": best,
                "weights": self.weights,
                "expected_clips_per_weights": expected_clips,
            },
        )
        self.rebuild(update)

    def error(self, update: int, attempt: int, failure: BaseException) -> None:
        write_json(
            self.directory
            / "errors"
            / f"update_{update:09d}_attempt_{attempt:03d}.json",
            {
                "update": update,
                "attempt": attempt,
                "status": "evaluation_error",
                "error": f"{type(failure).__name__}: {failure}",
            },
        )
        self.rebuild(update)

    def rebuild(self, restored_update: int) -> None:
        rounds = {
            row["update"]: row
            for item in (self.directory / "rounds").glob("*.json")
            if (row := json.loads(item.read_text()))["update"] <= restored_update
        }
        errors = {
            row["update"]
            for item in (self.directory / "errors").glob("*.json")
            if (row := json.loads(item.read_text()))["update"] <= restored_update
        }
        state = {
            "last_update": 0,
            "best": None,
            "bad_evaluations": 0,
            "stop_requested": False,
            "stop_reason": None,
            "policy": self.policy,
        }
        for update in sorted(set(rounds) | errors):
            if update != state["last_update"] + self.interval or update in errors:
                state["bad_evaluations"] = 0
            state["last_update"] = update
            if update not in rounds:
                continue
            candidate = rounds[update]["best"]
            if candidate is not None and (
                state["best"] is None or candidate["score"] < state["best"]["score"]
            ):
                state["best"] = candidate
                state["bad_evaluations"] = 0
            else:
                state["bad_evaluations"] += 1
        state["stop_requested"] = bool(
            self.policy["enabled"]
            and state["last_update"] >= self.policy["min_updates"]
            and state["bad_evaluations"] >= self.policy["patience_evaluations"]
        )
        if state["stop_requested"]:
            state["stop_reason"] = "physical_validation_plateau"
        self.state = state
        write_json(self.directory / "state.json", state)
        write_json(self.directory / "best_checkpoint.json", state["best"])
        self.export_candidates(restored_update)
