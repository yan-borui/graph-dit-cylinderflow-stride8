"""Immutable, dependency-free contract for the nine four-GPU attention runs."""

from __future__ import annotations

PROTOCOL = "graph_dit.attention_ablation4.uvp_c4.v1"
CHECKPOINT_FORMAT = "graph_dit.attention_ablation4.training.v1"
REPRESENTATION_ID = "d0fee50b-8a47-4652-b229-0e82e06ba2d7"
AE_REPOSITORY = "DingDong1921/cylinderflow-vgae-uvp-epoch1180"
AE_REVISION = "f21e74636a850c571f9870be91c6f420d41aa4d6"
AE_BYTES = 249943363
UPDATES = 62500
WINDOWS = 250000
VARIANTS = {
    "h1": ("graph_hop_mask", 1),
    "h2": ("graph_hop_mask", 2),
    "full": ("full", None),
}
TASKS = tuple(f"{variant}_seed{seed}" for seed in range(3) for variant in VARIANTS)
WEIGHTS = ["raw", "ema_0.999", "ema_0.9999"]


def is_ablation(config: dict) -> bool:
    return config.get("protocol") == PROTOCOL


def variant_name(config: dict) -> str:
    model = config["model"]
    if model["attention_mode"] == "graph_hop_mask":
        return f"H{model['graph_hop_limit']}"
    return model["attention_mode"].upper()


def validate_ablation(config: dict) -> None:
    """Lock scientific settings while allowing only attention and training seed."""
    variant = config.get("ablation", {}).get("variant")
    if variant not in VARIANTS or config.get("seed") not in (0, 1, 2):
        raise ValueError("ablation requires H1/H2/Full and training seeds 0/1/2")
    if "scaling" in config:
        raise ValueError("ablation and scaling protocols are separate")
    mode, hop = VARIANTS[variant]
    expected = {
        "model": {
            "attention_mode": mode,
            "graph_hop_limit": hop,
            "width": 512,
            "blocks": 24,
            "heads": 8,
            "mlp_ratio": 4.0,
            "future_frames": 64,
            "diffusion_steps": 1000,
            "latent_features": 4,
            "condition_features": 512,
        },
        "ablation": {"variant": variant, "parameter_count": 115013124},
        "representation": {
            "representation_id": REPRESENTATION_ID,
            "autoencoder_format": "vgae_cf.uvp_dit_autoencoder.v1",
            "vgae_config_id": "w512_d4-4-2_c4",
            "autoencoder_seed": 0,
            "autoencoder_mode": "formal",
        },
        "distributed": {"world_size": 4, "gradient_as_bucket_view": True},
        "window_schedule": {
            "budget_windows": WINDOWS,
            "total_windows": WINDOWS,
            "warmup_windows": 4000,
            "checkpoint_every_windows": 50000,
            "recovery_every_windows": 5000,
            "log_every_windows": 100,
            "validation_every_windows": 50000,
        },
        "validation": {
            "every_updates": 12500,
            "sampling_steps": 20,
            "sampling_seeds": [0, 1, 2],
            "weights": WEIGHTS,
            "selection": "validation24_complete_strict_uv_raw_ema",
            "early_stopping": {
                "enabled": False,
                "min_updates": 62500,
                "patience_evaluations": 20,
            },
        },
        "training": {
            "budget_updates": UPDATES,
            "effective_batch": 4,
            "microbatch": 1,
            "gradient_accumulation": 1,
            "learning_rate": 1e-4,
            "weight_decay": 1e-6,
            "gradient_clip": 1.0,
            "precision": "fp32",
            "schedule": "cosine",
            "warmup_updates": 1000,
            "schedule_total_updates": UPDATES,
            "min_learning_rate": 1e-7,
            "decay_start_updates": 31250,
            "decay_period_updates": 12500,
            "decay_factor": 0.1,
            "ema_decays": [0.999, 0.9999],
            "ema_decay_unit": "window",
            "activation_checkpointing": True,
            "checkpoint_every_updates": 12500,
            "recovery_every_updates": 1250,
            "log_every_updates": 25,
        },
    }
    for section, value in expected.items():
        if config.get(section) != value:
            raise ValueError(f"{section} differs from the locked ablation contract")
    if config.get("preflight_max_graph", False) not in (True, False):
        raise ValueError("preflight_max_graph must be boolean")
