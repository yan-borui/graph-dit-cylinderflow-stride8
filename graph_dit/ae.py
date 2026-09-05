"""Optional reproduction of the fixed VGAE recipe; prepare it once for a campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from dgn4cfd import DataLoader
from dgn4cfd.nn.models.vgae import VGAE
from .data import DATA_REVISION
from .representation import open_data
from .runtime import (
    acquire_run_lock,
    append_json,
    load_checkpoint,
    save_checkpoint,
    rng_state,
    restore_rng,
    seed_everything,
    write_json,
)
from .train import configure_runtime

ARCHITECTURE = {
    "in_node_features": 3,
    "cond_node_features": 4,
    "cond_edge_features": 2,
    "latent_node_features": 1,
    "depths": [2, 2, 1],
    "fnns_depth": 2,
    "fnns_width": 126,
    "aggr": "sum",
    "dropout": 0.1,
    "norm_latents": False,
}


def loss_components(model, graph, posterior_mean: bool = False) -> tuple:
    if posterior_mean:
        _, mean, logvar, c_list, e_list, edge_list, batch_list = model.encode(
            graph, graph.field
        )
        prediction = model.decode(
            graph,
            mean,
            c_list,
            e_list,
            edge_list,
            batch_list,
            graph.dirichlet_mask,
            graph.field[:, -3:],
        )
    else:
        prediction, _, mean, logvar = model(graph)
    reconstruction = F.mse_loss(prediction, graph.target)
    kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction + 1e-6 * kl, reconstruction, kl


class Frames(torch.utils.data.Dataset):
    def __init__(self, graphs, indices: tuple, epoch: int | None = None):
        self.graphs, self.indices, self.epoch = graphs, indices, epoch
        self.items = [(index, frame) for index in indices for frame in (0, 25, 50, 74)]

    def __len__(self):
        return len(self.indices) if self.epoch is not None else len(self.items)

    def __getitem__(self, ordinal):
        if self.epoch is None:
            index, frame = self.items[ordinal]
        else:
            index = self.indices[ordinal]
            frame = int(
                np.random.default_rng(
                    np.random.SeedSequence([0, self.epoch, index])
                ).integers(0, 75)
            )
        return self.graphs.get_sequence(index, sequence_start=frame, n_in=1)


def train_ae(
    data_dir: Path,
    output: Path,
    device_name: str,
    *,
    resume: bool = False,
    epochs: int = 5000,
    debug: bool = False,
) -> dict:
    data, graphs = open_data(data_dir, debug=debug)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    lock = acquire_run_lock(output)
    try:
        device = torch.device(device_name)
        configure_runtime(device, "fp32")
        seed_everything(0)
        model = VGAE(arch=ARCHITECTURE, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=50, eps=0.0
        )
        start, best, best_epoch, update = 1, float("inf"), None, 0
        identity = {
            "normalization": data.manifest["train_only_normalization"],
            "dataset_revision": DATA_REVISION,
            "data_identity": data.identity(),
            "arch": ARCHITECTURE,
            "debug": debug,
        }
        if resume:
            saved = load_checkpoint(output / "latest.pt")
            if json.dumps(saved["identity"], sort_keys=True) != json.dumps(
                identity, sort_keys=True
            ):
                raise ValueError("VGAE resume dataset or architecture differs")
            model.load_state_dict(saved["weights"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            restore_rng(saved["rng"])
            start, best, best_epoch, update = (
                saved["epoch"] + 1,
                saved["best"],
                saved["best_epoch"],
                saved["update"],
            )
        write_json(
            output / "recipe.json",
            {
                **identity,
                "seed": 0,
                "batch_size": 16,
                "lr": 1e-4,
                "max_epochs": epochs,
                "kl_weight": 1e-6,
                "validation_frames": [0, 25, 50, 74],
                "scheduler": "ReduceLROnPlateau, Train total, factor=0.1 patience=50, endpoint lr<1e-8",
            },
        )
        val_loader = DataLoader(
            Frames(graphs, data.splits["validation"]),
            batch_size=16,
            shuffle=False,
            num_workers=0,
        )
        started = time.perf_counter()
        for epoch in range(start, epochs + 1):
            loader = DataLoader(
                Frames(graphs, data.splits["train"], epoch),
                batch_size=16,
                shuffle=True,
                generator=torch.Generator().manual_seed(epoch),
                num_workers=0,
            )
            values = []
            model.train()
            for graph in loader:
                graph.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, reconstruction, kl = loss_components(model, graph)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite VGAE loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                update += 1
                values.append(
                    [float(value.detach()) for value in (loss, reconstruction, kl)]
                )
            total = np.mean(values, axis=0)
            scheduler.step(float(total[0]))
            validation = None
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                model.eval()
                with torch.no_grad():
                    validation = float(
                        np.mean(
                            [
                                float(loss_components(model, graph.to(device), True)[0])
                                for graph in val_loader
                            ]
                        )
                    )
                if validation < best:
                    best, best_epoch = validation, epoch
                    save_checkpoint(
                        output / "best.pt",
                        {
                            **identity,
                            "representation_id": str(uuid.uuid4()),
                            "weights": model.state_dict(),
                            "epoch": epoch,
                            "validation_total": best,
                        },
                    )
            row = {
                "epoch": epoch,
                "update": update,
                "train_total": float(total[0]),
                "validation_total": validation,
                "best_epoch": best_epoch,
                "best_validation_total": best,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            append_json(output / "training.jsonl", row)
            print(json.dumps(row), flush=True)
            save_checkpoint(
                output / "latest.pt",
                {
                    "identity": identity,
                    "weights": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": rng_state(),
                    "epoch": epoch,
                    "update": update,
                    "best": best,
                    "best_epoch": best_epoch,
                },
            )
            if optimizer.param_groups[0]["lr"] < 1e-8:
                break
        write_json(output / "summary.json", row)
        return row
    finally:
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_ae(args.data_dir, args.output_dir, args.device, resume=args.resume)


if __name__ == "__main__":
    main()
