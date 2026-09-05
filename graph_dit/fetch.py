"""Download only the two fixed-revision Train/Validation input files."""

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

from .data import DATA_REPOSITORY, DATA_REVISION, Dataset
from .runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    files = [
        "cylinderflow_stride8_75frames.h5",
        "cylinderflow_stride8_75frames_manifest.json",
    ]
    for name in files:
        folder = "data" if name.endswith(".h5") else "metadata"
        url = f"https://huggingface.co/datasets/{DATA_REPOSITORY}/resolve/{DATA_REVISION}/{folder}/{name}"
        temporary = args.output_dir / (name + ".partial")
        print(f"Downloading {name} from pinned revision {DATA_REVISION}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("xb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
        os.replace(temporary, args.output_dir / name)
    dataset = Dataset(args.output_dir / files[0], args.output_dir / files[1])
    write_json(args.output_dir / "download_identity.json", dataset.identity())
    print("Train/Validation contract verified", flush=True)


if __name__ == "__main__":
    main()
