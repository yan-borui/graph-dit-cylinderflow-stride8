"""Prepare Airfoil latents using an explicitly trained Airfoil VGAE export."""

from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--autoencoder", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from graph_dit.config import load_config
    from graph_dit.representation import prepare

    config = load_config(
        Path(__file__).parent / "configs/airfoil_h1_w512_d24_4gpu.json"
    )
    prepare(args.data_dir, args.autoencoder, args.artifacts, args.device, config=config)


if __name__ == "__main__":
    main()
