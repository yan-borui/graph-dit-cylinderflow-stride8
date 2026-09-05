"""Validate immutable scientific settings and evaluate the exact LR plan."""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import TRAINING_PROTOCOL


def load_config(file_name: str | Path) -> dict:
    config = json.loads(Path(file_name).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config.get("protocol") != TRAINING_PROTOCOL:
        raise ValueError("unsupported training protocol")
    model, training, validation = (
        config[key] for key in ("model", "training", "validation")
    )
    fixed = {
        "attention_mode": "graph_hop_mask",
        "graph_hop_limit": 1,
        "heads": 8,
        "mlp_ratio": 4.0,
        "future_frames": 64,
        "diffusion_steps": 1000,
        "latent_features": 1,
        "condition_features": 126,
    }
    if any(model.get(key) != value for key, value in fixed.items()):
        raise ValueError(
            "H1, eight heads, joint64 and the frozen representation must stay fixed"
        )
    if any(
        training.get(key) != 1
        for key in ("effective_batch", "microbatch", "gradient_accumulation")
    ):
        raise ValueError(
            "microbatch, accumulation and effective batch must all equal one"
        )
    if model["width"] < 8 or model["width"] % 8 or model["blocks"] < 1:
        raise ValueError(
            "width must be a positive multiple of eight; depth must be positive"
        )
    if training["schedule"] not in {"cosine", "late_decay", "constant"}:
        raise ValueError("unknown LR schedule")
    if training["precision"] not in {"fp32", "bf16"}:
        raise ValueError("supported training precision: fp32 or bf16")
    for key in ("learning_rate", "min_learning_rate", "gradient_clip", "decay_factor"):
        if not math.isfinite(training[key]) or training[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 < training["min_learning_rate"] <= training["learning_rate"]:
        raise ValueError("min LR must not exceed peak LR")
    if not 0 < training["decay_factor"] < 1:
        raise ValueError("decay_factor must be between zero and one")
    if not 0 <= training["warmup_updates"] < training["schedule_total_updates"]:
        raise ValueError("warmup must precede the fixed schedule endpoint")
    if training["decay_start_updates"] < training["warmup_updates"]:
        raise ValueError("late decay must follow warmup")
    for key in (
        "checkpoint_every_updates",
        "recovery_every_updates",
        "log_every_updates",
        "decay_period_updates",
    ):
        if not isinstance(training[key], int) or training[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not math.isfinite(training["weight_decay"]) or training["weight_decay"] < 0:
        raise ValueError("weight decay must be finite and nonnegative")
    decays = training["ema_decays"]
    if len(set(decays)) != len(decays) or any(not 0 < decay < 1 for decay in decays):
        raise ValueError("EMA decays must be unique and between zero and one")
    allowed = {"raw", *(f"ema_{decay:g}" for decay in decays)}
    if not validation["weights"] or not set(validation["weights"]) <= allowed:
        raise ValueError(
            "Validation weights must exist among raw and maintained EMA states"
        )
    if validation["sampling_seeds"] != [0, 1, 2] or validation["sampling_steps"] != 20:
        raise ValueError(
            "quality evaluation fixes 20 DDIM steps and sampling labels 0/1/2"
        )
    if validation["every_updates"] < 1 or config["seed"] < 0:
        raise ValueError("invalid Validation interval or training seed")


def learning_rate(config: dict, update: int) -> float:
    """LR for the upcoming, one-based optimizer update; never reset on resume."""
    training = config["training"]
    warmup, total = training["warmup_updates"], training["schedule_total_updates"]
    if not 1 <= update <= total:
        raise ValueError("optimizer update exceeds the immutable LR plan")
    peak, floor = training["learning_rate"], training["min_learning_rate"]
    if update <= warmup:
        return peak * update / warmup
    if training["schedule"] == "constant":
        return peak
    if training["schedule"] == "cosine":
        progress = (update - warmup) / (total - warmup)
        return floor + (peak - floor) * (1 + math.cos(math.pi * progress)) / 2
    count = max(
        0,
        (update - training["decay_start_updates"]) // training["decay_period_updates"]
        + 1,
    )
    return max(floor, peak * training["decay_factor"] ** count)
