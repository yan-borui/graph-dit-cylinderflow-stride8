"""Run the same end-to-end FP32 cost protocol as the four published baselines."""

import argparse
import json
from pathlib import Path
import time

import torch

from .evaluate import Predictor, load_selected
from .ablation_contract import variant_name
from .performance import benchmark, configure
from .runtime import monitor_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    selected = json.loads((args.run / "selection.json").read_text())
    if not selected.get("complete_stage"):
        raise ValueError("benchmark requires a selected, completed stage")
    device = torch.device(args.device)
    configure(device)
    begin = time.perf_counter()
    model, checkpoint = load_selected(
        args.run / selected["checkpoint"], args.artifacts, selected["weights"]
    )
    model.to(device)
    predictor = Predictor(model, args.artifacts, args.data_dir, device, "fp32")
    load_seconds = time.perf_counter() - begin
    benchmark(
        method=f"graph_dit_{variant_name(checkpoint['config']).lower()}",
        indices=monitor_indices(predictor.data.splits["validation"]),
        load_case=predictor.load_case,
        predict=lambda sample: predictor.predict(sample, diagnostics=False)[0],
        device=device,
        output_dir=args.output_dir,
        data_identity=predictor.data.identity(),
        provenance={
            "checkpoint_id": checkpoint["checkpoint_id"],
            "weights": selected["weights"],
            "artifact_id": checkpoint["artifact_id"],
            "training_seed": checkpoint["config"]["seed"],
            "update": checkpoint["update"],
            "training_precision": checkpoint["config"]["training"]["precision"],
        },
        models=[model, predictor.codec.autoencoder],
        model_load_seconds=load_seconds,
    )


if __name__ == "__main__":
    main()
