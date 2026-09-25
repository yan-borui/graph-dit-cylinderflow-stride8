"""Selection, provenance and dataset checks for supplementary flow movies."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import node_area_weights


DATASETS = {
    "cylinderflow": ("cylinderflow.physical_prediction.v2", 0.08),
    "airfoil": ("airfoil.uvp.physical_prediction.v1", 0.0016),
}
PAIRED_KEYS = (
    "points",
    "cells",
    "node_type",
    "target",
    "raw_frame_indices",
    "physical_time",
)
MODEL_KEYS = ("checkpoint_id", "weights", "artifact_id", "training_seed")


def write_record(destination: Path, record: dict[str, Any]) -> None:
    """Publish a complete JSON record atomically within the output directory."""
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def resolve_file(root: Path, template: str, trajectory: int | None = None) -> Path:
    """Resolve a config-relative path with a single supported placeholder."""
    if trajectory is not None:
        if "{trajectory}" not in template:
            raise ValueError("prediction templates must contain {trajectory}")
        template = template.format(trajectory=trajectory)
    file_name = Path(template).expanduser()
    return (root / file_name).resolve()


def paired(reference: dict, candidate: dict) -> None:
    """Require the same geometry, ordering, truth and evaluation times."""
    for key in PAIRED_KEYS:
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"paired prediction differs in {key}")


def load_bundle(source: Path, dataset: str, trajectory: int) -> tuple[dict, dict]:
    """Read a physical UVP archive without loading a model or modifying fields."""
    with np.load(source, allow_pickle=False) as archive:
        required = set(PAIRED_KEYS) | {
            "prediction",
            "trajectory_index",
            "seed",
            "units",
            "prediction_schema",
            "provenance",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{source}: missing keys {sorted(missing)}")
        bundle = {key: archive[key] for key in required}
        if "count" in archive:
            bundle["count"] = archive["count"]
    schema, dt = DATASETS[dataset]
    if str(bundle["prediction_schema"]) != schema:
        raise ValueError(f"{source}: expected schema {schema}")
    if str(bundle["units"]) != "physical_uvp":
        raise ValueError(f"{source}: expected physical_uvp units")
    if int(bundle["trajectory_index"]) != trajectory or trajectory not in range(
        1000, 1100
    ):
        raise ValueError(f"{source}: expected Validation trajectory {trajectory}")
    points, cells = bundle["points"], bundle["cells"]
    nodes = len(points)
    if points.shape != (nodes, 2) or not np.isfinite(points).all():
        raise ValueError(f"{source}: invalid [N,2] mesh points")
    if (
        cells.ndim != 2
        or cells.shape[1] != 3
        or cells.size == 0
        or cells.dtype.kind not in "iu"
        or cells.min() < 0
        or cells.max() >= nodes
    ):
        raise ValueError(f"{source}: invalid triangle connectivity")
    node_area_weights(points, cells)
    if bundle["node_type"].shape != (nodes,):
        raise ValueError(f"{source}: node_type must preserve the N node labels")
    for key in ("target", "prediction"):
        if bundle[key].shape != (65, nodes, 3) or not np.isfinite(bundle[key]).all():
            raise ValueError(f"{source}: {key} must be finite [65,N,3]")
    if not np.array_equal(bundle["prediction"][0], bundle["target"][0]):
        raise ValueError(f"{source}: observed frame zero differs from target")
    if not np.array_equal(bundle["raw_frame_indices"], np.arange(65) * 8):
        raise ValueError(f"{source}: expected raw frames 0,8,...,512")
    if bundle["physical_time"].shape != (65,) or not np.allclose(
        bundle["physical_time"],
        np.arange(65) * dt,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(f"{source}: expected stored-frame dt={dt}")
    provenance = json.loads(str(bundle["provenance"]))
    if not isinstance(provenance, dict) or not provenance.get("checkpoint_id"):
        raise ValueError(f"{source}: checkpoint provenance is required")
    return bundle, provenance


def identity(source: Path, bundle: dict, provenance: dict) -> dict:
    """Record file identity and existing experiment provenance without hashes."""
    stat = source.stat()
    return {
        "file": str(source),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sampling_label": int(bundle["seed"]),
        "provenance": provenance,
    }


def model_binding(provenance: dict) -> dict:
    """Bind a movie to a fixed model, representation and inference recipe."""
    keys = MODEL_KEYS + (
        "ae_checkpoint_id",
        "representation_id",
        "normalization",
        "sampling_steps",
        "sampling_schedule",
        "config",
        "configuration",
    )
    return {key: provenance[key] for key in keys if key in provenance}


def load_method(
    method: dict, root: Path, dataset: str, trajectory: int
) -> tuple[dict, dict]:
    """Prefer an existing K8 mean, otherwise average the eight fixed members."""
    if method["sampling"] == "single":
        source = resolve_file(root, method["template"], trajectory)
        bundle, provenance = load_bundle(source, dataset, trajectory)
        if int(bundle["seed"]) != 0:
            raise ValueError(f"{source}: single prediction requires sampling label 0")
        if int(provenance.get("ensemble_size", bundle.get("count", 1))) != 1:
            raise ValueError(f"{source}: single prediction points to an ensemble")
        if (
            "training_seed" in method
            and provenance.get("training_seed") != method["training_seed"]
        ):
            raise ValueError(
                f"{source}: expected training seed {method['training_seed']}"
            )
        if method["name"] in {"H1", "H2", "Full"}:
            model = provenance.get("config", provenance.get("configuration", {})).get(
                "model", {}
            )
            expected = "full" if method["name"] == "Full" else "graph_hop_mask"
            if model.get("attention_mode") != expected or (
                method["name"] != "Full"
                and model.get("graph_hop_limit") != int(method["name"][1])
            ):
                raise ValueError(
                    f"{source}: attention configuration differs from {method['name']}"
                )
        return bundle, {
            "name": method["name"],
            "sampling": "single",
            "sampling_label": 0,
            "binding": model_binding(provenance),
            "sources": [identity(source, bundle, provenance)],
        }

    mean_file = (
        resolve_file(root, method["mean_template"], trajectory)
        if method.get("mean_template")
        else None
    )
    if mean_file is not None and mean_file.exists():
        bundle, provenance = load_bundle(mean_file, dataset, trajectory)
        seeds = provenance.get("member_prng_seeds", [])
        if (
            provenance.get("ensemble_size") != 8
            or provenance.get("aggregation") != "physical_uvp_mean"
            or len(seeds) != 8
            or len(set(seeds)) != 8
            or ("count" in bundle and int(bundle["count"]) != 8)
        ):
            raise ValueError(
                f"{mean_file}: expected physical K8 mean and eight member seeds"
            )
        return bundle, {
            "name": method["name"],
            "sampling": "mean_k8",
            "group": method["group"],
            "binding": model_binding(provenance),
            "member_prng_seeds": seeds,
            "sources": [identity(mean_file, bundle, provenance)],
        }

    members = method.get("members", [])
    if len(members) != 8:
        raise FileNotFoundError(
            f"K8 mean missing: {mean_file}; provide all eight members"
        )
    sources = [resolve_file(root, item["template"], trajectory) for item in members]
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError("missing K8 members: " + ", ".join(missing))
    if len(set(sources)) != 8:
        raise ValueError("K8 members must be eight distinct files")
    reference = None
    mean = None
    binding = None
    records, seeds = [], []
    for number, (item, source) in enumerate(zip(members, sources), 1):
        bundle, provenance = load_bundle(source, dataset, trajectory)
        if int(bundle["seed"]) != item["seed"]:
            raise ValueError(f"{source}: sampling label differs from configured member")
        if int(provenance.get("ensemble_size", bundle.get("count", 1))) != 1:
            raise ValueError(f"{source}: expected an individual K8 member")
        seed = provenance.get("prng_seed", provenance.get("sample_seed"))
        if seed is None:
            raise ValueError(f"{source}: member PRNG seed is required")
        seeds.append(seed)
        if reference is None:
            reference = bundle
            binding = model_binding(provenance)
            mean = np.zeros_like(bundle["prediction"], dtype=np.float64)
        else:
            paired(reference, bundle)
            if model_binding(provenance) != binding:
                raise ValueError("K8 members have different model/inference identities")
        # Match sampling_ensemble_support: online float64 mean, then float32 field.
        mean += (bundle["prediction"].astype(np.float64) - mean) / number
        records.append(identity(source, bundle, provenance))
    if len(set(seeds)) != 8:
        raise ValueError("K8 members must have eight distinct PRNG seeds")
    mean_provenance = {
        **binding,
        "ensemble_size": 8,
        "aggregation": "physical_uvp_mean",
        "member_prng_seeds": seeds,
        "export_group": method["group"],
    }
    reference = dict(
        reference,
        prediction=mean.astype(np.float32),
        seed=np.asarray(0),
        count=np.asarray(8),
        provenance=np.asarray(json.dumps(mean_provenance)),
    )
    if not np.array_equal(reference["prediction"][0], reference["target"][0]):
        raise ValueError("K8 float32 mean changed the observed initial field")
    return reference, {
        "name": method["name"],
        "sampling": "mean_k8",
        "group": method["group"],
        "binding": binding,
        "member_prng_seeds": seeds,
        "sources": records,
    }


def uv_score(bundle: dict) -> float:
    """Use the evaluator's area-weighted UV relative RMSE over future frames 1..64."""
    weights = node_area_weights(bundle["points"], bundle["cells"])[None, :, None]
    target = bundle["target"][1:, :, :2].astype(np.float64)
    error = bundle["prediction"][1:, :, :2].astype(np.float64) - target
    numerator = float(np.sum(weights * error**2))
    denominator = float(np.sum(weights * target**2))
    score = (
        math.sqrt(numerator / denominator)
        if denominator > 0
        else (0.0 if numerator == 0 else math.inf)
    )
    if not math.isfinite(score):
        raise ValueError("UV relative RMSE is nonfinite")
    return score


def validate_config(config: dict) -> None:
    """Constrain the three handoff tasks and their sampling conventions."""
    if config.get("dataset") not in DATASETS:
        raise ValueError("dataset must be cylinderflow or airfoil")
    task = config.get("task")
    expected = (
        {"GLaDiT", "MGN", "EAGLE", "AROMA", "Text2PDE"}
        if task == "comparison"
        else {"H1", "H2", "Full"}
        if task == "attention"
        else set()
    )
    methods = config.get("methods", [])
    names = [item["name"] for item in methods]
    if not expected or set(names) != expected or len(names) != len(expected):
        raise ValueError("provide each required task method exactly once")
    if task == "attention" and (
        config["dataset"] != "cylinderflow" or not config.get("selection_file")
    ):
        raise ValueError("attention requires CylinderFlow comparison selection_file")
    for method in methods:
        sampling = "mean_k8" if method["name"] == "GLaDiT" else "single"
        if method.get("sampling") != sampling:
            raise ValueError(f"{method['name']}: expected {sampling}")
        if sampling == "mean_k8":
            if not method.get("group"):
                raise ValueError("fix the GLaDiT sampling group explicitly")
            members = method.get("members", [])
            if members and (
                len(members) != 8 or len({m["seed"] for m in members}) != 8
            ):
                raise ValueError("provide eight distinct member sampling labels")
            if not method.get("mean_template") and not members:
                raise ValueError("provide a K8 mean template or eight members")
        elif not method.get("template"):
            raise ValueError("single prediction requires a template")
        if task == "attention" and method.get("training_seed") != 0:
            raise ValueError("attention comparisons require training seed 0")


def select_case(config: dict, root: Path, output: Path, explicit: int | None) -> dict:
    """Persist the chosen case before resolving any comparison method."""
    dataset = config["dataset"]
    if config["task"] == "attention":
        source = resolve_file(root, config["selection_file"])
        record = json.loads(source.read_text(encoding="utf-8"))
        if (
            record.get("state") != "selected"
            or record.get("dataset") != dataset
            or record.get("selector") != "GLaDiT K8"
            or record.get("task") != "comparison"
        ):
            raise ValueError(
                "selection_file must be a selected CylinderFlow GLaDiT K8 case"
            )
        trajectory = record["trajectory_index"]
        if explicit is not None and explicit != trajectory:
            raise ValueError(
                "attention must reuse the CylinderFlow comparison trajectory"
            )
        if trajectory not in range(1000, 1100):
            raise ValueError("selection_file is outside Validation")
        record = dict(record, task="attention", reused_from=str(source))
        write_record(output / "selection.json", record)
        return record

    method = next(item for item in config["methods"] if item["name"] == "GLaDiT")
    record = {
        "dataset": dataset,
        "task": "comparison",
        "state": "selecting",
        "selector": "GLaDiT K8",
        "selection_mode": "explicit" if explicit is not None else "best",
        "group": method["group"],
        "metric": "uv_relative_rmse",
        "metric_definition": "sqrt(sum(area_weight * UV_error^2) / sum(area_weight * GT_UV^2)), frames 1..64",
        "tie_break": "smallest trajectory index",
        "candidates": [],
        "failures": [],
    }
    binding = None
    best_source = None
    best_pair = (math.inf, math.inf)
    indices = [explicit] if explicit is not None else list(range(1000, 1100))
    for index in indices:
        try:
            bundle, source = load_method(method, root, dataset, index)
            if binding is None:
                binding = source["binding"]
            elif source["binding"] != binding:
                raise ValueError(
                    "selection candidates have different model/inference identities"
                )
            score = uv_score(bundle)
            record["candidates"].append(
                {
                    "trajectory_index": index,
                    "uv_relative_rmse": score,
                    "member_prng_seeds": source["member_prng_seeds"],
                    "files": [
                        {
                            key: value
                            for key, value in item.items()
                            if key != "provenance"
                        }
                        for item in source["sources"]
                    ],
                }
            )
            record["binding"] = binding
            if (score, index) < best_pair:
                best_pair, best_source = (score, index), source
        except (ValueError, OSError, KeyError, TypeError) as error:
            record["failures"].append({"trajectory_index": index, "error": str(error)})
        write_record(output / "selection.json", record)
        print(
            f"Selection {len(record['candidates']) + len(record['failures'])}/{len(indices)}",
            flush=True,
        )
    if record["failures"]:
        record["state"] = "incomplete"
        write_record(output / "selection.json", record)
        raise ValueError(
            "incomplete GLaDiT selection cohort; see selection.json failures"
        )
    record.update(
        state="selected",
        trajectory_index=best_pair[1],
        uv_relative_rmse=best_pair[0],
        selected_source=best_source,
    )
    write_record(output / "selection.json", record)
    return record


def export(config: dict, root: Path, output: Path, args: argparse.Namespace) -> None:
    """Resolve the selected paired fields, then render both media formats."""
    selection = select_case(config, root, output, args.trajectory)
    index = selection["trajectory_index"]
    bundles, records, failures = [], [], []
    for method in config["methods"]:
        try:
            bundle, source = load_method(method, root, config["dataset"], index)
            if bundles:
                paired(bundles[0], bundle)
            if method["name"] == "GLaDiT" and source != selection["selected_source"]:
                raise ValueError("selected GLaDiT source changed after selection")
            bundles.append(bundle)
            records.append(source)
        except (ValueError, OSError, KeyError, TypeError) as error:
            failures.append({"method": method["name"], "error": str(error)})
    write_record(output / "sources.json", {"methods": records, "failures": failures})
    if failures:
        raise ValueError(
            "selected case inputs failed:\n"
            + "\n".join(f"{item['method']}: {item['error']}" for item in failures)
        )
    inputs = output / "inputs"
    inputs.mkdir()
    files, labels = [], []
    for bundle, source in zip(bundles, records):
        destination = inputs / f"{source['name']}.npz"
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **bundle)
        files.append(destination)
        label = f"{source['name']} | " + (
            "K8 mean" if source["sampling"] == "mean_k8" else "sample 0"
        )
        if config["task"] == "attention":
            label += " | train seed 0"
        labels.append(label)
    # Import rendering after selection to preserve actionable missing-input records.
    from .media import render

    case_label = (
        "GLaDiT best Validation case"
        if selection["selection_mode"] == "best"
        else "Explicit Validation case"
    )
    title = f"{config['dataset']} | {case_label} {index} | selection: GLaDiT K8"
    if config["task"] == "attention":
        title += " | attention ablation"
    result = render(
        files,
        output / "media",
        labels=labels,
        fps=args.fps,
        snapshots=[],
        title=title,
        paired_rows=True,
        frames=args.frames,
    )
    write_record(
        output / "status.json",
        {
            "state": "complete",
            "dataset": config["dataset"],
            "task": config["task"],
            "trajectory_index": index,
            "fps": args.fps,
            "render": result,
        },
    )


def main() -> None:
    """Run one configured comparison with durable selection and failure records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory", type=int, choices=range(1000, 1100), metavar="1000..1099"
    )
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        help="Reduced render workload; default all 65 frames.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("fps must be finite and positive")
    if args.frames is not None and (
        len(set(args.frames)) != len(args.frames)
        or any(i < 0 or i > 64 for i in args.frames)
    ):
        parser.error("frames must be distinct values in 0..64")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("use a new output directory")
    output.mkdir(parents=True)
    try:
        config_file = args.config.resolve(strict=True)
        config = json.loads(config_file.read_text(encoding="utf-8"))
        validate_config(config)
        write_record(output / "config.json", config)
        write_record(output / "status.json", {"state": "running"})
        export(config, config_file.parent, output, args)
    except Exception as error:
        write_record(
            output / "status.json",
            {
                "state": "failed",
                "error": str(error),
                "exception": type(error).__name__,
            },
        )
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
