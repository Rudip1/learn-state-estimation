#!/usr/bin/env python3
"""Generate the reader's exercise notebooks from the solution notebooks.

In a solution notebook, the part of a code cell the reader has to write is enclosed in

    ### BEGIN SOLUTION
    ...
    ### END SOLUTION

In the exercise copy that block is replaced by a placeholder, and all outputs and execution counts are
removed. Everything else (text, plots, assert cells, break-it cells) is copied unchanged.

    python tools/make_exercises.py            # all notebooks
    python tools/make_exercises.py 03 04      # only the given chapter numbers
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
SOLUTIONS = ROOT / "2_notebooks" / "solutions"
EXERCISES = ROOT / "2_notebooks" / "exercises"

BLOCK = re.compile(r"^([ \t]*)### BEGIN SOLUTION\n.*?^[ \t]*### END SOLUTION[ \t]*\n?", re.S | re.M)


def strip_solution(source: str) -> str:
    def placeholder(match: re.Match) -> str:
        indent = match.group(1)
        return f'{indent}# ✏️ your code here\n{indent}raise NotImplementedError("exercise")\n'

    return BLOCK.sub(placeholder, source)


def convert(src: Path, dst: Path) -> None:
    nb = nbformat.read(src, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = strip_solution(cell.source)
            cell.outputs = []
            cell.execution_count = None
        cell.metadata.pop("execution", None)
    nb.metadata.pop("widgets", None)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, dst)


def main(argv: list[str]) -> int:
    for src in sorted(SOLUTIONS.glob("*.ipynb")):
        if argv and not any(src.name.startswith(a) for a in argv):
            continue
        convert(src, EXERCISES / src.name)
        print(f"wrote {(EXERCISES / src.name).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
