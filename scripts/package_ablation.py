"""Create a source-only handoff archive; inspect member names and byte lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = [
        "LICENSE",
        "pyproject.toml",
        "cylinderflow_upstream.json",
        "ABLATION_4GPU.md",
        "STATIC_CHECKS.json",
        "scripts/ablation_4gpu.sh",
        "scripts/slurm_ablation_4gpu.sh",
        "scripts/package_ablation.py",
    ]
    names.extend(
        str(item.relative_to(root)).replace("\\", "/")
        for folder, pattern in (
            ("graph_dit", "*.py"),
            ("dgn4cfd", "*.py"),
            ("configs/attention_ablation_4gpu", "*.json"),
        )
        for item in (root / folder).rglob(pattern)
    )
    names = sorted(set(names))
    if any(not (root / name).is_file() for name in names):
        raise ValueError("required handoff file is missing")
    manifest = {
        "base_commit": "4901f33",
        "branch": "feature/attention-ablation-4gpu",
        "validation": "static checks only; target four-GPU execution pending",
        "members": [
            {"name": name, "bytes": (root / name).stat().st_size} for name in names
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prefix = "graph-dit-attention-ablation-4gpu/"
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(root / name, prefix + name)
        archive.writestr(prefix + "README.md", (root / "ABLATION_4GPU.md").read_bytes())
        archive.writestr(
            prefix + "PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n"
        )
    with zipfile.ZipFile(args.output) as archive:
        actual = archive.namelist()
        if len(actual) != len(names) + 2 or len(actual) != len(set(actual)):
            raise ValueError("archive member count differs from the source manifest")
        for name in names:
            if archive.read(prefix + name) != (root / name).read_bytes():
                raise ValueError(f"archive bytes differ from source: {name}")
    print(
        json.dumps(
            {
                "archive": str(args.output.resolve()),
                "files": len(names) + 2,
                "bytes": args.output.stat().st_size,
                "verified": "member names and exact byte content",
            }
        )
    )


if __name__ == "__main__":
    main()
