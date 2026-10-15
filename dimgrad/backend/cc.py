"""Compile generated C with GCC and run it (used by the tests and the evaluation scripts)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_gcc():
    return shutil.which("gcc")


def compile_c(c_text: str, workdir, name: str = "prog", flags=("-O2",), werror: bool = True) -> Path:
    """Write `c_text` to workdir/name.c and build it. Raises RuntimeError with GCC's output on failure."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / f"{name}.c"
    exe = workdir / f"{name}.exe"
    source.write_text(c_text, encoding="utf-8")
    result = subprocess.run(
        ["gcc", "-std=c11", "-Wall", *(["-Werror"] if werror else []), *flags,
         "-o", str(exe), str(source), "-lm"],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gcc failed:\n{result.stderr}")
    return exe


def run_queries(exe, queries) -> list[list[float]]:
    """Run harness queries [(function name, [args])]; one list of floats per query."""
    text = "".join(f"{name} {' '.join(repr(float(a)) for a in args)}\n" for name, args in queries)
    result = subprocess.run([str(exe)], input=text, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"program failed ({result.returncode}): {result.stderr}")
    return [[float(x) for x in line.split()] for line in result.stdout.splitlines()]
