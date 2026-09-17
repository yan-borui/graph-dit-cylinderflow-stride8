"""Use the shared local Airfoil converter instead of CylinderFlow downloads."""

from pathlib import Path
from .data import Dataset


def fetch_data(output_dir: Path) -> Dataset:
    return Dataset(
        output_dir / "airfoil_stride8_75frames.h5",
        output_dir / "airfoil_stride8_75frames_manifest.json",
    )


def main() -> None:
    raise SystemExit(
        "Prepare Airfoil with: python -m airfoil_data.prepare --raw-dir RAW --output-dir DATA_DIR"
    )


if __name__ == "__main__":
    main()
