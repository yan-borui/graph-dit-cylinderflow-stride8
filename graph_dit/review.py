"""Re-score or render any of the five methods' shared physical NPZ archives."""

import argparse
from pathlib import Path

from .media import render
from .score import score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "render"))
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--every", type=int, default=1)
    args = parser.parse_args()
    if args.command == "score":
        score(args.inputs, args.output_dir)
    else:
        render(args.inputs, args.output_dir, labels=args.labels, every=args.every)


if __name__ == "__main__":
    main()
