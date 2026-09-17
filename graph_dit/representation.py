"""Prepare one frozen VGAE/Train cache shared by every independent DiT run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch

from dgn4cfd.cylinderflow_data import (
    CylinderFlowH5Dataset,
    CylinderFlowNormalization,
    build_cylinderflow_transform,
)
from dgn4cfd.graph_distance import unweighted_shortest_path_hops
from dgn4cfd.nn.diffusion.graph_window_codec import FrozenUVPLatentCodec
from .data import Dataset, DATA_REPOSITORY, DATA_REVISION
from .config import load_config
from .runtime import seed_everything, write_json

ARTIFACT_FORMAT = "graph_dit.ae75.train_cache.v1"
PERSONAL_AE_FORMAT = "vgae_cf.airfoil_uvp_dit_autoencoder.v1"
AE_ASSET = "vgae_stride8_epoch930.pt"
AE_REPRESENTATION_ID = "cylinderflow_stride8_vgae_epoch930_release_v1"
AE_REPOSITORY = "DingDong1921/graph-dit-cylinderflow-stride8"
AE_REVISION = "8abcd6128a9896d6097d5dccb60cd0beaa4b852e"
AE_RELEASE = (
    "https://github.com/yan-borui/graph-dit-cylinderflow-stride8/releases/download/representation-v1/"
    + AE_ASSET
)
AE_SOURCES = {
    "huggingface": (
        f"https://huggingface.co/{AE_REPOSITORY}/resolve/{AE_REVISION}/{AE_ASSET}"
    ),
    "github": AE_RELEASE,
}


def paths(data_dir: Path) -> tuple[Path, Path]:
    return (
        data_dir / "airfoil_stride8_75frames.h5",
        data_dir / "airfoil_stride8_75frames_manifest.json",
    )


def open_data(
    data_dir: Path, *, debug: bool = False
) -> tuple[Dataset, CylinderFlowH5Dataset]:
    dataset_file, manifest_file = paths(data_dir)
    data = Dataset(dataset_file, manifest_file, debug=debug)
    graphs = CylinderFlowH5Dataset(
        dataset_file,
        manifest_path=manifest_file,
        transform=build_cylinderflow_transform(manifest_file),
    )
    return data, graphs


def fetch_autoencoder(output: Path, *, source: str = "huggingface") -> None:
    raise RuntimeError(
        "Airfoil requires a new Airfoil VGAE export; no CylinderFlow weights are downloaded"
    )


def checkpoint_features(checkpoint: dict) -> dict:
    """Read representation dimensions from either supported UVP weight export."""
    arch = checkpoint.get("arch", {})
    if arch.get("in_node_features") != 3:
        raise ValueError("the frozen autoencoder must encode and decode UVP")
    dimensions = {
        "latent_features": arch.get("latent_node_features"),
        "condition_features": arch.get("fnns_width"),
    }
    for name, value in dimensions.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"invalid autoencoder {name}")
    if checkpoint.get("format") == PERSONAL_AE_FORMAT:
        metadata = checkpoint.get("metadata", {})
        if (
            metadata.get("fields") != ["u", "v", "p"]
            or metadata.get("architecture") != arch
            or metadata.get("latent_channels") != dimensions["latent_features"]
            or metadata.get("condition_features") != dimensions["condition_features"]
            or metadata.get("representation_id") != checkpoint.get("representation_id")
            or checkpoint.get("test_accessed") is not False
        ):
            raise ValueError("personal UVP export metadata is inconsistent")
    return dimensions


def validate_model_representation(model_config: dict, identity: dict) -> None:
    """Bind the DiT input/output widths to the actual frozen representation."""
    for name in ("latent_features", "condition_features"):
        if model_config.get(name) != identity.get(name):
            raise ValueError(f"DiT {name} differs from the frozen representation")


def validate_representation_config(config: dict, identity: dict) -> None:
    validate_model_representation(config["model"], identity)
    for name, value in config.get("representation", {}).items():
        if identity.get(name) != value:
            raise ValueError(f"the requested representation differs: {name}")


def prepare(
    data_dir: Path,
    autoencoder: Path,
    output: Path,
    device: str,
    *,
    debug: bool = False,
    config: dict | None = None,
) -> dict:
    data, graphs = open_data(data_dir, debug=debug)
    checkpoint = torch.load(autoencoder, map_location="cpu", weights_only=True)
    if not checkpoint.get("representation_id"):
        raise ValueError(
            "use an identified UVP weight export, including dit_autoencoder.pt"
        )
    if checkpoint.get("normalization") != data.manifest["train_only_normalization"]:
        raise ValueError("the AE was trained with a different physical normalizer")
    if checkpoint.get("dataset_revision") != DATA_REVISION and not debug:
        raise ValueError("AE dataset revision differs from the released data")
    features = checkpoint_features(checkpoint)
    representation = {
        **features,
        "autoencoder_architecture": checkpoint["arch"],
        "autoencoder_format": checkpoint.get("format"),
        "vgae_config_id": checkpoint.get("config_id"),
        "autoencoder_seed": checkpoint.get("seed"),
        "autoencoder_mode": checkpoint.get("mode"),
    }
    if config is not None:
        validate_representation_config(config, representation)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(0)
    started = time.perf_counter()
    codec = FrozenUVPLatentCodec(str(autoencoder), device=device)
    shutil.copyfile(autoencoder, output / "autoencoder.pt")
    shutil.copyfile(paths(data_dir)[1], output / "dataset_manifest.json")
    identity = {
        **representation,
        "format": ARTIFACT_FORMAT,
        "artifact_id": str(uuid.uuid4()),
        "representation_id": checkpoint["representation_id"],
        "dataset_repository": DATA_REPOSITORY,
        "dataset_revision": DATA_REVISION,
        "data_identity": data.identity(),
        "normalization": checkpoint["normalization"],
        "train_frames": [0, 74],
        "dit_train_frames": [0, 64],
        "debug": debug,
        "train_count": len(data.splits["train"]),
        "state": "preparing",
    }
    write_json(output / "artifact.json", identity)
    latent_sum = latent_square = None
    count = 0
    with h5py.File(output / "train_latents.partial.h5", "x") as cache:
        cache.attrs["format"] = ARTIFACT_FORMAT
        cache.attrs["artifact_id"] = identity["artifact_id"]
        cache.attrs["representation_id"] = identity["representation_id"]
        cache.create_dataset("sim_indices", data=data.splits["train"])
        for index in data.splits["train"]:
            graph = graphs.get_sequence(index, n_in=1)
            first, context = codec.encode_context(graph, graph.field)
            physical = data.read(index)["field"]
            normalization = CylinderFlowNormalization.from_manifest(paths(data_dir)[1])
            mean = torch.tensor(normalization.field_mean)
            std = torch.tensor(normalization.field_std)
            latent_frames = [first.cpu().numpy()]
            for frame in range(1, 75):
                scaled = (torch.from_numpy(physical[frame]) - mean) / std
                latent, *_ = codec.encode_field(graph, scaled)
                latent_frames.append(latent.cpu().numpy())
            latents = np.stack(latent_frames).astype(np.float32)
            if not np.isfinite(latents).all():
                raise FloatingPointError(
                    f"nonfinite latent cache at Train trajectory {index}"
                )
            group = cache.create_group(f"sim_{index:05d}")
            for name, values in (
                ("latents", latents),
                ("node_context", context.node_context.cpu().numpy()),
                ("positions", context.positions.cpu().numpy()),
                (
                    "graph_hops",
                    unweighted_shortest_path_hops(
                        context.edge_index.cpu(), num_nodes=context.num_latent_nodes
                    ),
                ),
            ):
                group.create_dataset(name, data=values)
            flattened = latents.astype(np.float64).reshape(-1, codec.latent_features)
            total, square = flattened.sum(0), (flattened**2).sum(0)
            latent_sum = total if latent_sum is None else latent_sum + total
            latent_square = square if latent_square is None else latent_square + square
            count += len(flattened)
            print(
                json.dumps(
                    {
                        "event": "cache",
                        "trajectory": index,
                        "latent_nodes": latents.shape[1],
                    }
                ),
                flush=True,
            )
        mean = latent_sum / count
        std = np.sqrt(np.maximum(latent_square / count - mean**2, 1e-12))
        cache.create_dataset("latent_mean", data=mean.astype(np.float32))
        cache.create_dataset("latent_std", data=std.astype(np.float32))
    os.replace(output / "train_latents.partial.h5", output / "train_latents.h5")
    identity.update(
        state="complete",
        elapsed_seconds=time.perf_counter() - started,
        latent_mean=mean.tolist(),
        latent_std=std.tolist(),
        latent_node_frames=count,
        autoencoder_parameters=sum(p.numel() for p in codec.parameters()),
    )
    write_json(output / "artifact.json", identity)
    return identity


def load_artifacts(
    directory: Path,
    data: Dataset | None = None,
    *,
    config: dict | None = None,
    autoencoder: Path | None = None,
) -> dict:
    identity = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
    if identity.get("format") != ARTIFACT_FORMAT or identity.get("state") != "complete":
        raise ValueError("representation preparation is incomplete or unsupported")
    ae = torch.load(directory / "autoencoder.pt", map_location="cpu", weights_only=True)
    if ae.get("representation_id") != identity["representation_id"]:
        raise ValueError("representation checkpoint ID does not match artifacts")
    if ae.get("normalization") != identity["normalization"]:
        raise ValueError("representation normalization mismatch")
    features = checkpoint_features(ae)
    representation = {
        **features,
        "autoencoder_architecture": ae["arch"],
        "autoencoder_format": ae.get("format"),
        "vgae_config_id": ae.get("config_id"),
        "autoencoder_seed": ae.get("seed"),
        "autoencoder_mode": ae.get("mode"),
    }
    for name, value in representation.items():
        if name in identity and identity[name] != value:
            raise ValueError(f"prepared representation metadata differs: {name}")
        identity[name] = value
    if config is not None:
        validate_representation_config(config, identity)
    if autoencoder is not None:
        requested = torch.load(autoencoder, map_location="cpu", weights_only=True)
        if (
            requested.get("representation_id") != identity["representation_id"]
            or requested.get("normalization") != identity["normalization"]
            or requested.get("dataset_revision") != ae.get("dataset_revision")
            or requested.get("arch") != ae["arch"]
        ):
            raise ValueError(
                "the explicitly requested autoencoder differs from the cache"
            )
    if data is not None:
        if json.dumps(identity["data_identity"], sort_keys=True) != json.dumps(
            data.identity(), sort_keys=True
        ):
            raise ValueError("dataset contract and representation dependencies differ")
        if identity["normalization"] != data.manifest["train_only_normalization"]:
            raise ValueError(
                "dataset normalization differs from the frozen representation"
            )
    with h5py.File(directory / "train_latents.h5", "r") as cache:
        if cache.attrs.get("artifact_id") != identity["artifact_id"]:
            raise ValueError("latent cache ID does not match artifacts")
        expected = identity["data_identity"]["split_indices"]["train"]
        if list(cache["sim_indices"]) != list(expected):
            raise ValueError("latent cache is not exactly the declared Train split")
        if cache.attrs.get("representation_id") != identity["representation_id"]:
            raise ValueError("cache and autoencoder dependencies differ")
        channels = features["latent_features"]
        for name in ("latent_mean", "latent_std"):
            values = np.asarray(identity[name], dtype=np.float32)
            if (
                values.shape != (channels,)
                or not np.isfinite(values).all()
                or (name == "latent_std" and np.any(values <= 0))
                or not np.array_equal(cache[name][:], values)
            ):
                raise ValueError(f"invalid or inconsistent Train {name}")
        for index in expected:
            group = cache[f"sim_{index:05d}"]
            latent_shape = group["latents"].shape
            if (
                len(latent_shape) != 3
                or latent_shape[0] != 75
                or latent_shape[2] != channels
            ):
                raise ValueError(
                    f"Train latent dimensions differ at trajectory {index}"
                )
            nodes = latent_shape[1]
            if (
                group["node_context"].shape != (nodes, features["condition_features"])
                or group["positions"].shape != (nodes, 2)
                or group["graph_hops"].shape != (nodes, nodes)
            ):
                raise ValueError(
                    f"Train graph context dimensions differ at trajectory {index}"
                )
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-ae")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--source", choices=tuple(AE_SOURCES), default="huggingface")
    build = commands.add_parser("prepare")
    build.add_argument("--data-dir", type=Path, required=True)
    build.add_argument("--autoencoder", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--device", default="cuda:0")
    build.add_argument(
        "--config", type=Path, help="bind preparation to this DiT configuration"
    )
    verify = commands.add_parser(
        "verify", help="check data, weights and prepared cache identity"
    )
    verify.add_argument("--data-dir", type=Path, required=True)
    verify.add_argument("--autoencoder", type=Path, required=True)
    verify.add_argument("--artifacts", type=Path, required=True)
    verify.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fetch-ae":
        fetch_autoencoder(args.output, source=args.source)
    elif args.command == "verify":
        identity = load_artifacts(
            args.artifacts,
            Dataset(*paths(args.data_dir)),
            config=load_config(args.config),
            autoencoder=args.autoencoder,
        )
        print(
            json.dumps(
                {
                    key: identity[key]
                    for key in (
                        "artifact_id",
                        "representation_id",
                        "latent_features",
                        "condition_features",
                    )
                }
            )
        )
    else:
        prepare(
            args.data_dir,
            args.autoencoder,
            args.output_dir,
            args.device,
            config=load_config(args.config) if args.config else None,
        )


if __name__ == "__main__":
    main()
