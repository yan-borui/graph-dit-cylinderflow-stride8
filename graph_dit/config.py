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
    }
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
    if not isinstance(training.get("activation_checkpointing", False), bool):
        raise ValueError("activation_checkpointing must be boolean")
    if not isinstance(
        config.get("distributed", {}).get("gradient_as_bucket_view", False), bool
    ):
        raise ValueError("gradient_as_bucket_view must be boolean")
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
    if "scaling" in config:
        validate_scaling(config)


def validate_scaling(config: dict) -> None:
    """Keep the four-size experiment on one fixed scientific recipe."""
    model, training = config["model"], config["training"]
    sizes = {(512, 24), (768, 24), (1024, 24), (1024, 32)}
    width, depth = model["width"], model["blocks"]
    if (width, depth) not in sizes or config["seed"] not in (0, 1, 2):
        raise ValueError(
            "scaling uses the four declared sizes and training seeds 0/1/2"
        )
    expected_parameters = (18 * depth + 5) * width**2 + (15 * depth + 531) * width + 4
    if config["scaling"] != {
        "format": "graph_dit.scaling32.uvp_c4.v1",
        "model_id": f"w{width}_d{depth}",
        "parameter_count": expected_parameters,
    }:
        raise ValueError("scaling model identity or parameter count differs")
    if config.get("distributed") != {
        "world_size": 32,
        "gradient_as_bucket_view": True,
        "required_accelerator": "NVIDIA L20",
    }:
        raise ValueError("scaling fixes 32 L20 ranks and shared gradient buckets")
    expected_training = {
        "budget_updates": 125000,
        "effective_batch": 32,
        "microbatch": 1,
        "gradient_accumulation": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "gradient_clip": 1.0,
        "precision": "fp32",
        "schedule": "cosine",
        "warmup_updates": 4000,
        "schedule_total_updates": 125000,
        "min_learning_rate": 1e-7,
        "ema_decays": [0.999, 0.9999],
        "ema_decay_unit": "update",
        "activation_checkpointing": True,
        "checkpoint_every_updates": 5000,
        "recovery_every_updates": 1000,
        "log_every_updates": 100,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("scaling sizes must share the fixed training recipe")
    if model["latent_features"] != 4 or model["condition_features"] != 512:
        raise ValueError("scaling fixes the UVP-c4 representation dimensions")
    if config.get("representation", {}).get("representation_id") != (
        "d0fee50b-8a47-4652-b229-0e82e06ba2d7"
    ):
        raise ValueError("scaling requires the selected epoch1180 UVP representation")
    validation = config["validation"]
    if (
        validation["every_updates"] != 5000
        or validation["weights"] != ["raw", "ema_0.999", "ema_0.9999"]
        or validation.get("selection") != "validation24_complete_strict_uv_raw_ema"
        or validation.get("early_stopping", {}).get("enabled") is not False
    ):
        raise ValueError(
            "scaling keeps common physical validation and a fixed endpoint"
        )


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
