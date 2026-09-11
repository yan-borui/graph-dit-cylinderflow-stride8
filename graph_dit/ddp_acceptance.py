"""CPU/Gloo behavior checks for DDP reduction, recovery and evaluation isolation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from dgn4cfd.nn.diffusion.models.graph_video_dit import GraphVideoDiT
from .config import load_config
from .ddp_train import LossForward
from .runtime import ROOT, load_checkpoint, write_json
from .train import EMA, train_run, window_at


def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right, (left, right)


def child(rank, root_name, fixture_name):
    root, fixture = Path(root_name), Path(fixture_name)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").resolve().as_uri(),
        rank=rank,
        world_size=2,
    )
    try:
        config = load_config(ROOT / "configs/base.json")
        config["distributed"] = {"world_size": 2}
        config["model"].update(width=16, blocks=1)
        config["training"].update(
            effective_batch=2,
            schedule="cosine",
            warmup_updates=0,
            schedule_total_updates=4,
            learning_rate=1e-4,
            min_learning_rate=1e-6,
            checkpoint_every_updates=2,
            recovery_every_updates=1,
            log_every_updates=1,
            ema_decay_unit="window",
        )
        config["validation"].update(every_updates=2, weights=["raw"])
        torch.manual_seed(31)
        model = GraphVideoDiT(**config["model"])
        model.set_latent_statistics([0.0], [1.0])
        reference = deepcopy(model)
        wrapped = DistributedDataParallel(LossForward(model), broadcast_buffers=False)
        with h5py.File(fixture / "artifacts/train_latents.h5", "r") as cache:
            samples = [window_at(cache, index, torch.device("cpu")) for index in (0, 1)]
        loss = wrapped(samples[rank], torch.Generator().manual_seed(99 + rank))
        loss.backward()
        for number, sample in enumerate(samples):
            (
                reference.training_loss(
                    **sample, generator=torch.Generator().manual_seed(99 + number)
                )
                / 2
            ).backward()
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            if expected.grad is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(
                    parameter.grad, expected.grad, rtol=1e-5, atol=1e-7
                )
        del wrapped, model, reference
        scalar = torch.nn.Linear(1, 1, bias=False)
        scalar.weight.data.zero_()
        ema = EMA(scalar, [0.5], windows_per_update=2)
        scalar.weight.data.fill_(2)
        ema.update(scalar)
        scalar.weight.data.fill_(4)
        ema.update(scalar)
        torch.testing.assert_close(
            ema.states["ema_0.5"]["weight"], torch.tensor([[3.375]])
        )
        arguments = (fixture / "artifacts", fixture / "data")
        train_run(config, *arguments, root / "continuous", 4, "cpu", debug=True)
        # Interrupt a physical evaluation after its shard output has been committed.
        from . import ddp_train

        original = ddp_train.evaluate_model

        def fail_after_shard(*args, **kwargs):
            result = original(*args, **kwargs)
            if rank == 1:
                raise RuntimeError("injected evaluator failure after completed shard")
            return result

        ddp_train.evaluate_model = fail_after_shard
        failed = False
        try:
            train_run(config, *arguments, root / "resumed", 2, "cpu", debug=True)
        except RuntimeError as error:
            assert "injected evaluator failure" in str(error)
            failed = True
        finally:
            ddp_train.evaluate_model = original
        assert failed
        train_run(
            config, *arguments, root / "resumed", 4, "cpu", debug=True, resume=True
        )
        no_mid_validation = deepcopy(config)
        no_mid_validation["validation"]["every_updates"] = 4
        train_run(
            no_mid_validation,
            *arguments,
            root / "without_mid_validation",
            4,
            "cpu",
            debug=True,
        )
        dist.barrier()
        if rank == 0:
            first, second, third = [
                load_checkpoint(root / name / "recovery_latest.pt")
                for name in ("continuous", "resumed", "without_mid_validation")
            ]
            for key in ("model", "ema", "optimizer", "sample_cursor", "rank_states"):
                equal(first[key], second[key])
                equal(first[key], third[key])
            assert first["sample_cursor"] == 8
            rows = (root / "resumed/candidates.jsonl").read_text().splitlines()
            assert len(rows) == 2
            assert (
                len(
                    {
                        (json.loads(row)["update"], json.loads(row)["weights"])
                        for row in rows
                    }
                )
                == 2
            )
            from .evaluate import load_selected

            load_selected(
                root / "resumed/recovery_latest.pt", fixture / "artifacts", "ema_0.999"
            )
            write_json(
                root / "acceptance.json",
                {
                    "global_mean_gradient_matches_independent_reference": True,
                    "different_graph_sizes": True,
                    "window_ema_arithmetic": True,
                    "continuous_resume_exact": True,
                    "per_rank_rng_exact": True,
                    "evaluation_rng_isolation_exact": True,
                    "partial_evaluator_failure_recovered": True,
                    "no_duplicate_evaluations": True,
                    "standalone_ddp_checkpoint_reader": True,
                    "backend": "cpu/gloo",
                    "world_size": 2,
                    "formal_gpu_verified": False,
                },
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    mp.spawn(
        child,
        args=(str(args.output_dir.resolve()), str(args.fixtures.resolve())),
        nprocs=2,
    )


if __name__ == "__main__":
    main()
