"""Four-card environment binding and restoration evidence for attention ablations."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import h5py
import torch

from .runtime import write_json
from .distributed import Context
from .performance import runtime_identity


def gpu_topology() -> list[list[str]]:
    """Map visible CUDA UUIDs to the physical link matrix reported by NVIDIA."""
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    gpu_names = {}
    for line in query.splitlines():
        index, gpu_uuid = [item.strip() for item in line.split(",", 1)]
        gpu_names[gpu_uuid.removeprefix("GPU-").lower()] = f"GPU{index}"
    visible = []
    for index in range(4):
        gpu_uuid = (
            str(torch.cuda.get_device_properties(index).uuid)
            .removeprefix("GPU-")
            .lower()
        )
        if gpu_uuid not in gpu_names:
            raise ValueError("cannot map CUDA device UUID to physical GPU topology")
        visible.append(gpu_names[gpu_uuid])
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=True
    ).stdout
    lines = [line.split() for line in topology.splitlines() if line.strip()]
    header = next(
        line for line in lines if line[0].startswith("GPU") and "X" not in line
    )
    rows = {line[0]: line[1:] for line in lines if line[0] in visible and "X" in line}
    return [
        [rows[source][header.index(destination)] for destination in visible]
        for source in visible
    ]


def environment_signature(devices: list[dict]) -> dict:
    """Require one Linux/NCCL node with four identical >=48 GB CUDA devices."""
    if platform.system() != "Linux" or len(devices) != 4:
        raise ValueError("attention ablation requires one Linux node and four GPUs")
    if {item["rank"] for item in devices} != set(range(4)):
        raise ValueError("four distinct global ranks are required")
    if len({item["hostname"] for item in devices}) != 1:
        raise ValueError("attention ablation uses a single node")
    if {item["local_rank"] for item in devices} != set(range(4)):
        raise ValueError("local ranks must be 0,1,2,3")
    if any(
        item["backend"] != "nccl" or item["local_world_size"] != 4 for item in devices
    ):
        raise ValueError("all four ranks must use NCCL on one node")
    if len({(item["gpu"], item["total_memory_bytes"]) for item in devices}) != 1:
        raise ValueError("all four cards must have the same model and memory capacity")
    if any(item["total_memory_bytes"] < 48_000_000_000 for item in devices):
        raise ValueError("the configured experiment requires >=48 GB cards")
    if torch.cuda.device_count() != 4:
        raise ValueError("expose exactly the four allocated GPUs")
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    inference_environment = runtime_identity(
        torch.device("cuda", torch.cuda.current_device())
    )
    nccl_version = torch.cuda.nccl.version()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "gpu": devices[0]["gpu"],
        "memory_bytes": devices[0]["total_memory_bytes"],
        "world_size": 4,
        "nodes": 1,
        "backend": "nccl",
        "precision": "fp32",
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": list(nccl_version) if isinstance(nccl_version, tuple) else nccl_version,
        "hdf5": h5py.version.hdf5_version,
        "driver": sorted(set(driver)),
        "gpu_topology": gpu_topology(),
        "cpu_model": inference_environment["cpu_model"],
        "cpu_logical_count": inference_environment["cpu_logical_count"],
        "packages": {
            name: version(name)
            for name in ("numpy", "scipy", "h5py", "torch-geometric")
        },
        "peer_access": [
            [
                torch.cuda.can_device_access_peer(i, j) if i != j else True
                for j in range(4)
            ]
            for i in range(4)
        ],
    }


def bind_environment(
    config: dict, devices: list[dict], run: Path, ctx: Context
) -> dict:
    """Bind every task and its acceptance to the same cohort environment."""
    import fcntl
    import json

    signatures = ctx.all_call(lambda: environment_signature(devices))
    if any(item != signatures[0] for item in signatures):
        raise ValueError("rank runtime versions or device topology differ")
    signature = signatures[0]

    def bind() -> None:
        # Both cohort/runs/task and cohort/preflight/task share this parent.
        cohort = run.parent.parent
        cohort.mkdir(parents=True, exist_ok=True)
        with (cohort / ".environment.lock").open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            receipt = cohort / "environment.json"
            if receipt.exists():
                if json.loads(receipt.read_text()) != signature:
                    raise ValueError(
                        "cohort environment differs; use the original four-card environment"
                    )
            else:
                write_json(receipt, signature)

    ctx.primary_call(bind)
    return signature


def state_equal(left: Any, right: Any) -> bool:
    """Compare restored state exactly without changing RNG or executing a model."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            state_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def bind_inputs(identity: dict, run: Path, ctx: Context) -> None:
    """Freeze one source snapshot and cache/data identity for the whole campaign."""
    import fcntl
    import json

    from .train import freeze_source

    def bind() -> None:
        cohort = run.parent.parent
        receipt = {
            key: identity[key]
            for key in (
                "artifact_id",
                "representation_id",
                "data_identity",
                "normalization",
            )
        }
        with (cohort / ".inputs.lock").open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            destination = cohort / "inputs.json"
            if destination.exists() and json.loads(destination.read_text()) != receipt:
                raise ValueError(
                    "all nine tasks must share the same frozen representation/cache/data"
                )
            freeze_source(cohort, resume=(cohort / "source").exists())
            write_json(destination, receipt)

    ctx.primary_call(bind)
