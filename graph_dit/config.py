"""Validate immutable scientific settings and evaluate the exact LR plan."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

from . import TRAINING_PROTOCOL
from .physical_monitor import validate_policy


def load_config(file_name: str | Path) -> dict:
    config = json.loads(Path(file_name).read_text(encoding="utf-8"))
    config = resolve_window_config(config)
    validate_config(config)
    return config


def resolve_window_config(config: dict) -> dict:
    """Resolve an explicit global-window clock without reinterpreting old updates."""
    config = deepcopy(config)
    clock = config.get("window_schedule")
    if clock is None:
        return config
    world = config.get("distributed", {}).get("world_size", 1)
    if not isinstance(world, int) or isinstance(world, bool) or world < 1:
        raise ValueError("world_size must be a positive integer")
    training = config["training"]
    batch = world * training["microbatch"] * training["gradient_accumulation"]
    mappings = {
        "budget_windows": (training, "budget_updates"),
        "total_windows": (training, "schedule_total_updates"),
        "warmup_windows": (training, "warmup_updates"),
        "checkpoint_every_windows": (training, "checkpoint_every_updates"),
        "recovery_every_windows": (training, "recovery_every_updates"),
        "log_every_windows": (training, "log_every_updates"),
        "validation_every_windows": (config["validation"], "every_updates"),
    }
    for name, (destination, key) in mappings.items():
        value = clock[name]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value % batch
        ):
            raise ValueError(f"{name} must be divisible by effective batch {batch}")
        resolved = value // batch
        if config.get("window_schedule_resolved") and destination.get(key) != resolved:
            raise ValueError(f"resolved {key} conflicts with {name}")
        destination[key] = resolved
    training["effective_batch"] = batch
    if not 0 < training["budget_updates"] <= training["schedule_total_updates"]:
        raise ValueError("window budget must lie within the cosine horizon")
    config["window_schedule_resolved"] = True
    return config


def validate_config(config: dict) -> None:
    ablation_protocol = "graph_dit.airfoil.attention.ae75.joint65.locked.v1"
    ablation = config.get("protocol") == ablation_protocol
    if ablation:
        variants = {
            "h1": ("graph_hop_mask", 1),
            "h2": ("graph_hop_mask", 2),
            "full": ("full", None),
        }
        variant = config.get("attention_ablation")
        if variant not in variants:
            raise ValueError("attention ablation requires h1, h2, or full")
        baseline_file = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "airfoil_h1_w512_d24_4gpu.json"
        )
        expected = resolve_window_config(
            json.loads(baseline_file.read_text(encoding="utf-8"))
        )
        expected["protocol"] = ablation_protocol
        expected["attention_ablation"] = variant
        mode, hops = variants[variant]
        expected["model"].update(attention_mode=mode, graph_hop_limit=hops)
        if config != expected:
            raise ValueError(
                "Airfoil attention ablation changes only attention "
                "from the fixed four-GPU seed0 recipe"
            )
    elif config.get("protocol") != TRAINING_PROTOCOL or "attention_ablation" in config:
        raise ValueError("unsupported training protocol")
    model, training, validation = (
        config[key] for key in ("model", "training", "validation")
    )
    fixed = {
        "heads": 8,
        "mlp_ratio": 4.0,
        "future_frames": 64,
        "diffusion_steps": 1000,
    }
    if not ablation:
        fixed.update(attention_mode="graph_hop_mask", graph_hop_limit=1)
    if any(model.get(key) != value for key, value in fixed.items()):
        raise ValueError("H1, eight heads and joint64 must stay fixed")
    for name in ("latent_features", "condition_features"):
        value = model.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"model.{name} must be a positive integer")
    world = config.get("distributed", {}).get("world_size", 1)
    if not isinstance(world, int) or isinstance(world, bool) or world < 1:
        raise ValueError("world_size must be a positive integer")
    if (
        training.get("microbatch") != 1
        or training.get("gradient_accumulation") != 1
        or training.get("effective_batch") != world
    ):
        raise ValueError(
            "H1 requires one window per rank and effective_batch == world_size"
        )
    if config.get("window_schedule") is not None:
        if config != resolve_window_config(config):
            raise ValueError("resolve the window schedule before training")
    if training.get("ema_decay_unit", "update") not in {"update", "window"}:
        raise ValueError("EMA decay unit must be update or window")
    if world > 1 and validation.get("early_stopping", {}).get("enabled", False):
        raise ValueError("the distributed screening protocol disables early stopping")
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
    validate_policy(validation)


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
