"""Download the pinned public inference package using Python's standard library."""

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

REPOSITORY = "DingDong1921/gladit-cylinderflow-550k-ema09999"
REVISION = "31183037b16ab2e6dba7122dce019c1dabd6297a"
FILES = {
    "dit_ema.pt": 460136748,
    "artifacts/autoencoder.pt": 249882334,
    "artifacts/artifact.json": 19499,
    "artifacts/dataset_manifest.json": 427521,
    "inference_manifest.json": 2964,
    "LICENSE": 11357,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name, expected in FILES.items():
        target = args.output_dir / name
        if target.exists():
            if target.stat().st_size != expected:
                raise ValueError(f"Existing file size differs: {target}")
            print(f"Reusing {target}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            partial.open("wb") as stream,
        ):
            while block := response.read(1024 * 1024):
                stream.write(block)
        if partial.stat().st_size != expected:
            raise ValueError(f"Download size differs: {partial}")
        partial.rename(target)
        print(f"Downloaded {name}", flush=True)


if __name__ == "__main__":
    main()
