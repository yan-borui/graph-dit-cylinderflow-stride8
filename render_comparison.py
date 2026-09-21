"""Render the paper's fixed Validation cases from existing physical predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graph_dit.media import render
from graph_dit.predictions import validate_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        nargs=2,
        action="append",
        required=True,
        metavar=("LABEL", "NPZ_TEMPLATE"),
        help="Repeat per method; template contains {trajectory} and fixes one sample.",
    )
    parser.add_argument("--cases", type=int, nargs="+", default=[1095, 1013, 1086])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases) or any(
        index < 1000 or index >= 1100 for index in args.cases
    ):
        parser.error("cases must be distinct CylinderFlow Validation IDs 1000..1099")
    labels = [label for label, _ in args.method]
    if len(set(labels)) != len(labels):
        parser.error("method labels must be distinct")
    if args.output_dir.exists():
        parser.error("use a new output directory")
    records = []
    for index in args.cases:
        inputs, sources = [], []
        reference = None
        for label, template in args.method:
            if "{trajectory}" not in template:
                parser.error("each template must contain {trajectory}")
            source = Path(template.format(trajectory=index)).resolve(strict=True)
            with np.load(source, allow_pickle=False) as archive:
                bundle = {key: archive[key] for key in archive.files}
            validate_prediction(bundle)
            if str(bundle.get("units", "")) != "physical_uvp":
                raise ValueError(f"{source}: expected physical_uvp units")
            if (
                str(bundle.get("prediction_schema", ""))
                != "cylinderflow.physical_prediction.v2"
            ):
                raise ValueError(f"{source}: expected CylinderFlow prediction v2")
            if int(bundle["trajectory_index"]) != index:
                raise ValueError(f"{source}: trajectory differs from requested case")
            if int(bundle["seed"]) != 0:
                raise ValueError(
                    f"{source}: paper comparison requires sampling label 0"
                )
            if not np.isfinite(bundle["prediction"]).all():
                raise ValueError(
                    f"{source}: retain and report the nonfinite prediction"
                )
            if reference is not None:
                for key in (
                    "points",
                    "cells",
                    "target",
                    "physical_time",
                    "raw_frame_indices",
                ):
                    if not np.array_equal(bundle[key], reference[key]):
                        raise ValueError(f"{source}: paired {key} differs")
            reference = bundle
            inputs.append(source)
            sources.append(
                {
                    "label": label,
                    "file": str(source),
                    "seed": int(bundle["seed"]),
                    "provenance": json.loads(str(bundle["provenance"])),
                }
            )
        records.append((index, inputs, sources))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for index, inputs, sources in records:
        destination = args.output_dir / f"trajectory_{index}"
        render(
            inputs,
            destination,
            labels=labels,
            snapshots=[16, 32, 48, 64],
            snapshot_pdf=True,
        )
        (destination / "sources.json").write_text(
            json.dumps({"trajectory_index": index, "sources": sources}, indent=2),
            encoding="utf-8",
        )
    (args.output_dir / "handoff.json").write_text(
        json.dumps(
            {
                "dataset": "CylinderFlow",
                "split": "validation",
                "cases": args.cases,
                "labels": labels,
                "snapshots": [16, 32, 48, 64],
                "movie_frames": list(range(65)),
                "physical_dt": 0.08,
                "sampling": "one explicitly selected archived realization per method",
                "complete": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
