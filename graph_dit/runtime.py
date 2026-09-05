"""Portable configuration, progress, checkpoint, and timing primitives."""

from __future__ import annotations

import contextlib
import csv
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(file_name: Path, payload: dict) -> None:
    file_name.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_name.with_name(file_name.name + ".tmp")
    temporary.write_text(
        json.dumps(clean_json(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, file_name)


def append_json(file_name: Path, payload: dict) -> None:
    with file_name.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(clean_json(payload), sort_keys=True) + "\n")


def read_jsonl(file_name: Path) -> list[dict]:
    if not file_name.exists():
        return []
    return [
        json.loads(line)
        for line in file_name.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(file_name: Path, rows: list[dict]) -> None:
    columns = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
    )
    with file_name.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            [{key: clean_json(value) for key, value in row.items()} for row in rows]
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_checkpoint(file_name: str | Path) -> dict:
    # Checkpoints are trusted local training artifacts, never downloaded implicitly.
    return torch.load(file_name, map_location="cpu", weights_only=False)


def save_checkpoint(file_name: Path, payload: dict) -> None:
    temporary = file_name.with_name(file_name.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, file_name)


def code_identity() -> dict:
    def git(*arguments):
        result = subprocess.run(
            ["git", "-C", str(ROOT), *arguments], capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    return {
        "commit": git("rev-parse", "HEAD"),
        "working_tree": git("status", "--short"),
        "upstream": json.loads((ROOT / "cylinderflow_upstream.json").read_text()),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def measured(timings: dict, name: str, device: torch.device, enabled: bool = True):
    if not enabled:
        yield
        return
    synchronize(device)
    begin = time.perf_counter()
    yield
    synchronize(device)
    timings[name] = timings.get(name, 0.0) + time.perf_counter() - begin


def autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA")
    return torch.autocast(
        device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16
    )


def peak_memory(device: torch.device) -> dict:
    return (
        {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        if device.type == "cuda"
        else {}
    )


def monitor_indices(indices: tuple[int, ...]) -> tuple[int, ...]:
    positions = np.rint(np.linspace(0, len(indices) - 1, min(24, len(indices)))).astype(
        int
    )
    return tuple(indices[position] for position in positions)


def acquire_run_lock(directory: Path):
    """Keep an OS lock until training exits; a crash releases it automatically."""
    handle = (directory / ".run.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("this run directory already has an active training process")
    return handle


def sample_seed(training_seed: int, trajectory: int, sampling_seed: int) -> int:
    return int(
        np.random.SeedSequence(
            [training_seed, trajectory, sampling_seed]
        ).generate_state(1)[0]
    )
