"""Shared by the evaluation scripts and the tests: where the benchmarks are and their input grids."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
RESULTS_DIR = ROOT / "evaluation" / "results"


def benchmark_names() -> list[str]:
    return sorted(path.stem for path in BENCH_DIR.glob("*.dg"))


def load_source(name: str) -> str:
    return (BENCH_DIR / f"{name}.dg").read_text(encoding="utf-8")


def load_grids() -> dict:
    """benchmark -> function -> parameter -> list of values."""
    return json.loads((BENCH_DIR / "grids.json").read_text(encoding="utf-8"))


def grid_points(spec: dict, params: list[str]) -> list[list[float]]:
    """Every combination of the listed values, as argument lists in parameter order."""
    return [list(point) for point in itertools.product(*(spec[p] for p in params))]


def use_utf8_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def markdown_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)
