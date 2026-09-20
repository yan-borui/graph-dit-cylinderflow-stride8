"""Join fresh same-machine reports and plot quality versus measured latency."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_json(file_path: Path) -> dict:
    return json.loads(file_path.read_text(encoding="utf-8"))


def collect(
    roots: list[Path], campaign: str, metric: str
) -> tuple[list[dict], list[str]]:
    points = sorted(
        {file.resolve() for root in roots for file in root.rglob("point.json")}
    )
    rows, issues, seen = [], [], set()
    reference = None
    for file in points:
        point = read_json(file)
        if point.get("schema") != "cylinderflow.pareto_point.v1":
            continue
        if point.get("campaign_id") != campaign:
            raise ValueError(f"Different campaign: {file}")
        timing = read_json(file.parent / point["timing_file"])
        quality = read_json(file.parent / point["quality_file"])
        summary = quality.get("aggregate", quality)
        identity = (point["method"], point["sampling_steps"], point["ensemble_size"])
        if identity in seen:
            raise ValueError(f"Duplicate configuration: {identity}")
        seen.add(identity)
        if point.get("test_accessed") is not False:
            raise ValueError(f"Missing Test-access identity: {file}")
        for key in ("environment", "data_identity", "provenance"):
            if point[key] != timing[key]:
                raise ValueError(f"Timing/quality receipt differs: {key}: {file}")
        shared = {
            key: timing[key]
            for key in ("environment", "data_identity", "case_registry", "settings")
        }
        shared["evaluator"] = point["evaluator"]
        if reference is None:
            reference = shared
        else:
            differences = [key for key in shared if shared[key] != reference[key]]
            if differences:
                raise ValueError(f"Mixed comparison conditions {differences}: {file}")
        if quality.get("provenance") is not None:
            if (
                quality["provenance"]["checkpoint_id"]
                != point["provenance"]["checkpoint_id"]
            ):
                raise ValueError(f"Quality checkpoint mismatch: {file}")
        if quality.get("checkpoint_id") is not None:
            if quality["checkpoint_id"] != point["provenance"]["checkpoint_id"]:
                raise ValueError(f"Quality checkpoint mismatch: {file}")
        trajectories = summary.get("trajectory_metrics", [])
        expected_draws = 3 if point["method"] in ("aroma", "text2pde") else 1
        complete_quality = (
            summary.get("trajectory_count") == 100
            and summary.get("failed_clips") == 0
            and summary.get("clip_count") == 100 * expected_draws
            and {row["trajectory_index"] for row in trajectories}
            == set(range(1000, 1100))
            and all(
                row.get("finite") and row.get("sample_count") == expected_draws
                for row in trajectories
            )
        )
        if not (
            complete_quality
            and timing.get("formal")
            and timing.get("status") == "complete"
            and timing.get("complete_trajectories") == 24
        ):
            issues.append(f"Incomplete point excluded: {identity}")
            continue
        value = summary.get(metric, {}).get("mean")
        latency = timing["latency_seconds"]["mean"]
        if (
            value is None
            or not math.isfinite(value)
            or latency is None
            or not (0 < latency < math.inf)
        ):
            issues.append(f"Unavailable metric or latency excluded: {identity}")
            continue
        rows.append(
            {
                "method": identity[0],
                "sampling_steps": identity[1],
                "ensemble_size": identity[2],
                "latency_seconds": latency,
                "latency_median_seconds": timing["latency_seconds"]["median"],
                "latency_p90_seconds": timing["latency_seconds"]["p90"],
                "metric": metric,
                "quality": value,
                "quality_trajectories": 100,
                "peak_allocated_bytes": timing["cuda_peak_allocated_bytes"],
                "parameter_count": timing["model"]["parameter_count"],
                "checkpoint_id": point["provenance"]["checkpoint_id"],
                "weights": point["provenance"].get("weights", "selected"),
                "quality_semantics": point["quality_semantics"],
                "campaign_id": campaign,
                "source": str(file),
            }
        )
    expected = {
        ("graph_dit_h1", steps, count)
        for steps in (6, 20)
        for count in (1, 2, 4, 8, 16)
    }
    expected |= {
        ("mgn", None, 1),
        ("eagle", None, 1),
        ("aroma", 4, 1),
        ("text2pde", 20, 1),
    }
    actual = {
        (row["method"], row["sampling_steps"], row["ensemble_size"]) for row in rows
    }
    if actual - expected:
        raise ValueError(
            f"Configurations outside this figure's grid: {actual - expected}"
        )
    for missing in sorted(expected - actual, key=str):
        issues.append(f"Missing configuration: {missing}")
    for root in roots:
        for exit_file in root.rglob("exit.json"):
            if read_json(exit_file).get("exit_code") != 0:
                issues.append(f"Recorded failure: {exit_file}")
    return rows, issues


def render(
    rows: list[dict], output: Path, title: str, metric: str, partial: bool
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 5.2), layout="constrained")
    for steps, color, marker in ((6, "#0072B2", "o"), (20, "#D55E00", "s")):
        group = sorted(
            (
                row
                for row in rows
                if row["method"] == "graph_dit_h1" and row["sampling_steps"] == steps
            ),
            key=lambda row: row["ensemble_size"],
        )
        if not group:
            continue
        ax.plot(
            [row["latency_seconds"] for row in group],
            [row["quality"] for row in group],
            color=color,
            marker=marker,
            linewidth=1.8,
            label=f"GLaDiT, S={steps}",
        )
        for row in group:
            ax.annotate(
                f"K={row['ensemble_size']}",
                (row["latency_seconds"], row["quality"]),
                xytext=(4, 7 if steps == 6 else -13),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
    for method, label, marker, color in (
        ("mgn", "MeshGraphNets", "^", "#009E73"),
        ("eagle", "EAGLE", "D", "#CC79A7"),
        ("aroma", "AROMA", "P", "#555555"),
        ("text2pde", "Text2PDE", "X", "#B89600"),
    ):
        for row in rows:
            if row["method"] == method:
                ax.scatter(
                    row["latency_seconds"],
                    row["quality"],
                    s=85,
                    marker=marker,
                    color=color,
                    label=label,
                    zorder=3,
                )
    ax.set_xscale("log")
    ax.set_xlabel("End-to-end latency per 64-frame prediction (s)")
    ax.set_ylabel(
        "UV relative RMSE" if metric == "uv_relative_rmse" else metric.replace("_", " ")
    )
    ax.set_title(title + (" — PARTIAL" if partial else ""))
    ax.grid(True, which="both", alpha=0.2)
    ax.margins(x=0.12, y=0.18)
    ax.legend(fontsize=9)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"pareto.{extension}", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", default="uv_relative_rmse")
    parser.add_argument("--title", default="CylinderFlow: inference cost and quality")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        rows, issues = collect(args.inputs, args.campaign_id, args.metric)
        (args.output_dir / "issues.json").write_text(
            json.dumps(issues, indent=2), encoding="utf-8"
        )
        if not rows or (issues and not args.allow_partial):
            raise ValueError(
                "Incomplete campaign; see issues.json. No formal plot generated."
            )
        with (args.output_dir / "pareto.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if not args.collect_only:
            render(rows, args.output_dir, args.title, args.metric, bool(issues))
        print(f"Collected {len(rows)} points: {args.output_dir}")
    except Exception as error:
        (args.output_dir / "failure.json").write_text(
            json.dumps({"error": f"{type(error).__name__}: {error}"}, indent=2),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
