#!/usr/bin/env python3
"""Execute notebooks top to bottom and fail on the first error.

    python tools/run_notebooks.py 2_notebooks/solutions            # execute, do not modify files
    python tools/run_notebooks.py --inplace 2_notebooks/solutions  # also store the outputs

Accepts directories (every *.ipynb inside) and individual notebook files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def collect(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out.extend(sorted(p.glob("*.ipynb")) if p.is_dir() else [p])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--inplace", action="store_true", help="write executed outputs back to the files")
    parser.add_argument("--timeout", type=int, default=900, help="per-cell timeout in seconds")
    args = parser.parse_args()

    failures = 0
    for path in collect(args.paths):
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                                resources={"metadata": {"path": str(path.parent)}})
        t0 = time.time()
        try:
            client.execute()
        except CellExecutionError as err:
            failures += 1
            print(f"FAIL {path} ({time.time() - t0:.1f}s)\n{err}", file=sys.stderr)
            continue
        print(f"ok   {path} ({time.time() - t0:.1f}s)")
        if args.inplace:
            nbformat.write(nb, path)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
