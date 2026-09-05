"""Download CylinderFlow and the released VGAE, then prepare the shared Train cache."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/stride8"))
    parser.add_argument(
        "--autoencoder", type=Path, default=Path("artifacts/vgae_stride8_epoch930.pt")
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/shared"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ae-source", choices=("huggingface", "github"), default="huggingface"
    )
    parser.add_argument(
        "--download-only", action="store_true", help="download inputs without encoding"
    )
    args = parser.parse_args()

    from graph_dit.fetch import fetch_data
    from graph_dit.representation import (
        AE_REPRESENTATION_ID,
        fetch_autoencoder,
        load_artifacts,
        prepare,
    )

    data = fetch_data(args.data_dir)
    fetch_autoencoder(args.autoencoder, source=args.ae_source)
    if args.download_only:
        print("Data and VGAE are ready; rerun without --download-only to encode Train.")
        return

    if args.artifacts.exists():
        identity = load_artifacts(args.artifacts, data)
        if identity["representation_id"] != AE_REPRESENTATION_ID:
            raise ValueError("existing cache uses a different VGAE; choose --artifacts")
        print(f"Reusing completed Train cache: {args.artifacts}", flush=True)
    else:
        prepare(args.data_dir, args.autoencoder, args.artifacts, args.device)
    print(
        f"Ready: data={args.data_dir}, artifacts={args.artifacts}. "
        "Generate the search with: python -m graph_dit.campaign plan "
        "--output-dir campaigns/fp32_screen",
        flush=True,
    )


if __name__ == "__main__":
    main()
