"""Collect recorded Airfoil attention results without loading models or data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_record(file_name: Path) -> dict:
    """Read an available trainer record; propagate malformed-file errors."""
    if not file_name.is_file():
        return {}
    return json.loads(file_name.read_text(encoding="utf-8"))


def collect(cohort: Path) -> list[dict]:
    """Keep all three tasks visible, including incomplete and failed runs."""
    rows = []
    for variant in ("h1", "h2", "full"):
        run = cohort / f"{variant}_seed0"
        config = read_record(run / "config.json")
        status = read_record(run / "status.json")
        selected = read_record(run / "selection.json")
        expected = read_record(
            Path(__file__).resolve().parents[1]
            / "configs"
            / f"airfoil_ablation_{variant}_4gpu.json"
        )
        observed = dict(config)
        observed.pop("window_schedule_resolved", None)
        matched = bool(config) and observed == expected
        rows.append(
            {
                "attention": variant,
                "training_seed": config.get("seed"),
                "run": str(run),
                "config_matches": matched,
                "state": status.get(
                    "state", "pending" if not run.exists() else "incomplete"
                ),
                "optimizer_updates": status.get("update"),
                "budget_updates": 250000,
                "global_batch": 4,
                "selected_update": selected.get("update"),
                "selected_weights": selected.get("weights"),
                "validation24_uv_relative_rmse": selected.get("score")
                if matched
                else None,
                "failed_clips": selected.get("failed_clips"),
                "checkpoint": selected.get("checkpoint"),
                "artifact_id": selected.get("artifact_id"),
                "endpoint_raw_score": status.get("endpoint_scores", {}).get("raw")
                if matched
                else None,
                "endpoint_ema_0.999_score": status.get("endpoint_scores", {}).get(
                    "ema_0.999"
                )
                if matched
                else None,
                "endpoint_ema_0.9999_score": status.get("endpoint_scores", {}).get(
                    "ema_0.9999"
                )
                if matched
                else None,
                "selection_record": str(run / "selection.json"),
                "status_record": str(run / "status.json"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    args = parser.parse_args()
    cohort = args.cohort.resolve()
    if not cohort.is_dir():
        parser.error("cohort must be an existing campaign directory")
    rows = collect(cohort)
    payload = {
        "selection": "Original Validation-24 complete strict UV selection over raw and both EMA states",
        "sampling": "20 DDIM steps; independently score sampling seeds 0/1/2, then aggregate",
        "rows": rows,
    }
    (cohort / "attention_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (cohort / "attention_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(cohort / "attention_summary.csv")


if __name__ == "__main__":
    main()
