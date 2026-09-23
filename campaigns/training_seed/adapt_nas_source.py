"""Adapt the lock in an independent copy of the seed campaign's frozen source."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="independent source copy for a fresh training seed",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    runtime = source / "graph_dit/runtime.py"
    helper = Path(__file__).resolve().parents[2] / "graph_dit/portable_lock.py"
    if not (source / "graph_dit/preflight.py").is_file():
        parser.error(
            "the original production preflight.py must exist in the source copy"
        )
    if not helper.is_file():
        parser.error("run this adapter from the NAS delivery checkout")
    text = runtime.read_text(encoding="utf-8")
    functions = [
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef) and node.name == "acquire_run_lock"
    ]
    if len(functions) != 1:
        parser.error("expected one top-level acquire_run_lock function")
    function = functions[0]
    original = ast.get_source_segment(text, function)
    replacement = (
        "def acquire_run_lock(directory: Path):\n"
        '    """Exclude concurrent writers using an atomic shared-directory lock."""\n'
        "    from .portable_lock import DirectoryLock\n\n"
        '    return DirectoryLock(directory / ".run.lock").acquire()'
    )
    if original != replacement:
        if "fcntl.flock(" not in original or 'directory / ".run.lock"' not in original:
            parser.error(
                "unrecognized lock implementation; inspect this source before adapting it"
            )
        lines = text.splitlines(keepends=True)
        lines[function.lineno - 1 : function.end_lineno] = [replacement + "\n"]
        updated = "".join(lines)
        ast.parse(updated)
        backup = runtime.with_name("runtime.py.before_nas")
        if backup.exists():
            parser.error(
                "an adaptation backup already exists; inspect it before retrying"
            )
        shutil.copyfile(runtime, backup)
        runtime.write_text(updated, encoding="utf-8", newline="\n")
    destination = source / "graph_dit/portable_lock.py"
    if destination.resolve() != helper.resolve():
        shutil.copyfile(helper, destination)
    print(
        "Prepared the run lock in the source copy. Create a new run and acceptance directory."
    )


if __name__ == "__main__":
    main()
