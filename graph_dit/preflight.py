"""Target-device acceptance on the largest cached Train graph before dispatch."""

import argparse
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from dgn4cfd.nn.diffusion.models.graph_video_dit import GraphVideoDiT
from .config import load_config, learning_rate
from .evaluate import Predictor
from .representation import load_artifacts
from .runtime import autocast, peak_memory, seed_everything, synchronize, write_json
from .train import configure_runtime, EMA, window_at


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=8)
    args = parser.parse_args()
    if args.updates < 4:
        raise ValueError(
            "at least four updates are needed beyond AdaLN-Zero initialization"
        )
    config = load_config(args.config)
    if config.get("distributed", {}).get("world_size", 1) > 1:
        from .train import train_run

        config["preflight_max_graph"] = True
        train_run(
            config,
            args.artifacts,
            args.data_dir,
            args.output_dir,
            args.updates,
            args.device,
        )
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    identity = load_artifacts(args.artifacts)
    device = torch.device(args.device)
    configure_runtime(device, config["training"]["precision"])
    seed_everything(config["seed"])
    model = GraphVideoDiT(**config["model"]).to(device)
    model.set_latent_statistics(identity["latent_mean"], identity["latent_std"])
    optimizer = torch.optim.AdamW(
        model.parameters(), weight_decay=config["training"]["weight_decay"]
    )
    ema = EMA(model, config["training"]["ema_decays"])
    initial_parameters = {
        name: value.detach().cpu().clone() for name, value in model.named_parameters()
    }
    with h5py.File(args.artifacts / "train_latents.h5", "r") as cache:
        index = max(
            (int(value) for value in cache["sim_indices"]),
            key=lambda i: cache[f"sim_{i:05d}/latents"].shape[1],
        )
        sample = window_at(cache, index, device)
    records = []
    for update in range(1, args.updates + 1):
        synchronize(device)
        started = time.perf_counter()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(config, update)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, config["training"]["precision"]):
            loss = model.training_loss(**sample)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        attention_norm = float(model.blocks[0].attention.in_proj_weight.grad.norm())
        optimizer.step()
        ema.update(model)
        synchronize(device)
        records.append(
            {
                "update": update,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "attention_gradient_norm": attention_norm,
                "seconds": time.perf_counter() - started,
            }
        )
    if not any(row["attention_gradient_norm"] > 0 for row in records[2:]):
        raise RuntimeError(
            "preflight never reached a nonzero attention gradient after zero initialization"
        )
    parameter_changes = {}
    for name, state in {"raw": model.state_dict(), **ema.states}.items():
        changes = [
            float((state[key].detach().cpu() - value).abs().max())
            for key, value in initial_parameters.items()
        ]
        if not any(value > 0 for value in changes):
            raise RuntimeError(f"preflight did not update {name} parameters")
        if not all(torch.isfinite(state[key]).all() for key in initial_parameters):
            raise FloatingPointError(f"nonfinite {name} parameters")
        dtypes = sorted({str(state[key].dtype) for key in initial_parameters})
        if config["training"]["precision"] == "fp32" and dtypes != ["torch.float32"]:
            raise RuntimeError(f"unexpected {name} parameter precision: {dtypes}")
        parameter_changes[name] = {
            "max_absolute_change": max(changes),
            "parameter_dtypes": dtypes,
        }
    predictor = Predictor(
        model, args.artifacts, args.data_dir, device, config["training"]["precision"]
    )
    synchronize(device)
    prediction_started = time.perf_counter()
    prediction, _ = predictor.predict(predictor.load_case(index), 0)
    synchronize(device)
    prediction_seconds = time.perf_counter() - prediction_started
    if not np.isfinite(prediction).all() or prediction.shape[0] != 65:
        raise FloatingPointError("complete decoded 64-frame forecast is invalid")
    write_json(
        args.output_dir / "preflight.json",
        {
            "config": config,
            "train_trajectory": index,
            "latent_nodes": int(sample["positions"].shape[1]),
            "updates": records,
            "mean_update_seconds_after_warmup": float(
                np.mean([row["seconds"] for row in records[2:]])
            ),
            "forecast_shape": list(prediction.shape),
            "forecast_seconds": prediction_seconds,
            "parameter_changes": parameter_changes,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            **peak_memory(device),
            "claim": "bounded hardware/software acceptance; not convergence or quality evidence",
        },
    )


if __name__ == "__main__":
    main()
