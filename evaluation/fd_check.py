"""RQ1 (correctness): do the generated gradients match central finite differences?

For every gradient of every benchmark, on every point of its input grid, the
generated C (built with GCC) is compared with a central difference of the
generated C for the function itself. Reports the largest relative error.

    python -m evaluation.fd_check            # uses GCC
    python -m evaluation.fd_check --interp   # uses the built-in interpreter instead
"""
from __future__ import annotations

import argparse
import csv
import tempfile

from dimgrad import compile_source
from dimgrad.backend.cc import compile_c, find_gcc, run_queries
from dimgrad.backend.emit_c import emit_c
from dimgrad.glue.fd import TOLERANCE, relative_error, step_points
from dimgrad.middle.interp import evaluate

from .common import (RESULTS_DIR, benchmark_names, grid_points, load_grids, load_source,
                     markdown_table, use_utf8_output)


def gradient_errors(bench: str, grids: dict, use_c: bool, workdir) -> list[dict]:
    compiled = compile_source(load_source(bench))
    program = compiled.program
    queries, layout = [], []
    for grad in compiled.check.grads.values():
        spec = grids[bench][grad.func]
        for point in grid_points(spec, program.functions[grad.func].params):
            hi, lo, h = step_points(point, grad.index)
            queries += [(grad.name, point), (grad.func, hi), (grad.func, lo)]
            layout.append((grad, point, h))
    if use_c:
        exe = compile_c(emit_c(program, f"{bench}.dg", harness=True), workdir, bench)
        answers = run_queries(exe, queries)
    else:
        answers = []
        for name, args in queries:
            result = evaluate(program, name, args)
            answers.append(list(result) if isinstance(result, tuple) else [result])

    rows: dict[str, dict] = {}
    for i, (grad, point, h) in enumerate(layout):
        (ad, value), (f_hi,), (f_lo,) = answers[3 * i: 3 * i + 3]
        fd = (f_hi - f_lo) / (2 * h)
        error = relative_error(ad, fd, value, h)
        row = rows.setdefault(grad.name, {
            "benchmark": bench, "gradient": grad.name, "unit": program.fmt(grad.unit),
            "points": 0, "max_rel_error": 0.0, "worst_point": point})
        row["points"] += 1
        if error >= row["max_rel_error"]:
            row["max_rel_error"], row["worst_point"] = error, point
    return list(rows.values())


def main(argv=None) -> int:
    use_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interp", action="store_true", help="evaluate with the interpreter, not GCC")
    args = parser.parse_args(argv)
    use_c = not args.interp and find_gcc() is not None
    if not use_c and not args.interp:
        print("gcc not found: falling back to the interpreter")

    grids = load_grids()
    rows = []
    with tempfile.TemporaryDirectory() as workdir:
        for bench in benchmark_names():
            rows += gradient_errors(bench, grids, use_c, workdir)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rq1_finite_differences.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = markdown_table(
        ["benchmark", "gradient", "unit of the derivative", "points", "largest relative error", "within tolerance"],
        [[r["benchmark"], f"`{r['gradient']}`", r["unit"], r["points"], f"{r['max_rel_error']:.2e}",
          "yes" if r["max_rel_error"] < TOLERANCE else "NO"] for r in rows])
    worst = max(r["max_rel_error"] for r in rows)
    failed = [r for r in rows if r["max_rel_error"] >= TOLERANCE]
    summary = (f"{len(rows)} gradients from {len({r['benchmark'] for r in rows})} programs, "
               f"{sum(r['points'] for r in rows)} points, evaluated with "
               f"{'GCC-compiled C' if use_c else 'the interpreter'}. Tolerance {TOLERANCE:g}. "
               f"Largest relative error {worst:.2e}; {len(failed)} gradient(s) outside tolerance.")
    (RESULTS_DIR / "rq1_finite_differences.md").write_text(
        f"# RQ1: gradients against central finite differences\n\n{summary}\n\n{table}\n", encoding="utf-8")
    print(table)
    print()
    print(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
