"""RQ4 (cost): how expensive is the generated gradient, and what does the optimiser save?

Measures, for every gradient of every benchmark:
  - three-address instructions before and after our optimiser (static count);
  - run time of the gradient relative to the function, with GCC at -O2;
  - the same gradient emitted without our optimiser, at -O2 and at -O0, which
    separates what our passes save from what GCC would have removed anyway;
  - a central difference costs two function calls, so its relative cost is 2.
And, for gradients with a closed form, the accuracy of the generated gradient
and of a central difference against the exact value.

    python -m evaluation.bench
"""
from __future__ import annotations

import csv
import math
import subprocess
import tempfile

from dimgrad import compile_source
from dimgrad.backend.cc import compile_c, find_gcc, run_queries
from dimgrad.backend.emit_c import cname, emit_c
from dimgrad.glue.fd import step_points
from dimgrad.middle import tac

from .common import (RESULTS_DIR, benchmark_names, grid_points, load_grids, load_source,
                     markdown_table, use_utf8_output)

BATCH_SECONDS = 0.02
REPEATS = 5

# Closed forms, written independently of the benchmark programs.
G, GRAVITY, LBF = 6.6743e-11, 9.81, 4.4482216152605
EXACT = {
    "energy_grad_v": lambda mass, v: mass * v,
    "energy_grad_mass": lambda mass, v: 0.5 * v * v,
    "potential_grad_r": lambda r, m1, m2: G * m1 * m2 / r ** 2,
    "orbital_speed_grad_r": lambda r, central: -0.5 * math.sqrt(G * central / r) / r,
    "period_grad_len": lambda len: math.pi / math.sqrt(GRAVITY * len),
    "power_grad_v": lambda v, rho, area: 1.5 * rho * 0.3 * area * v * v,
    "top_speed_grad_p": lambda p, rho, area: (2 * p / (rho * 0.3 * area)) ** (1 / 3) / (3 * p),
    "total_energy_grad_x": lambda x, v, k, mass: k * x,
    "total_energy_grad_v": lambda x, v, k, mass: mass * v,
    "delta_v_grad_impulse": lambda impulse, mass: LBF / mass,
    "miss_grad_v": lambda v, angle, target: 2 * v * math.sin(2 * angle) / GRAVITY,
    "f_grad_x": lambda x: 50.0 * (x - 0.2),
    "g_grad_x": lambda x, a: 2 * x,
}


def timing_main(program: tac.Program, targets: list[tuple[str, list[float]]]) -> str:
    """A main() that prints `name nanoseconds-per-call` for each target."""
    out = [
        "#include <stdio.h>",
        "#include <time.h>",
        "",
        "static double now(void)",
        "{",
        "    struct timespec ts;",
        "    timespec_get(&ts, TIME_UTC);",
        "    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;",
        "}",
        "",
        "int main(void)",
        "{",
        "    double sink = 0.0, v = 0.0;",
    ]
    for name, args in targets:
        func = program.functions[name]
        names = [f"a{i}" for i in range(len(args))]
        call_args = names + (["&v"] if func.kind == "grad" else [])
        call = f"sink += {cname(name)}({', '.join(call_args)}) + v;"
        decls = ", ".join(f"{n} = {a!r}" for n, a in zip(names, args))
        out += [
            "    {",
            f"        volatile double {decls};",  # volatile: the calls cannot be folded away
            "        long n = 256;",
            "        double best = 1e30;",
            "        for (;;) {",
            "            double start = now();",
            f"            for (long i = 0; i < n; i++) {call}",
            f"            if (now() - start >= {BATCH_SECONDS}) break;",
            "            n *= 2;",
            "        }",
            f"        for (int rep = 0; rep < {REPEATS}; rep++) {{",
            "            double start = now();",
            f"            for (long i = 0; i < n; i++) {call}",
            "            double t = (now() - start) / (double)n;",
            "            if (t < best) best = t;",
            "        }",
            f"        printf(\"{name} %.6g\\n\", best * 1e9);",
            "    }",
        ]
    out += ["    fprintf(stderr, \"%g\\n\", sink);", "    return 0;", "}"]
    return "\n".join(out) + "\n"


def measure(program: tac.Program, targets, workdir, tag: str, flag: str) -> dict[str, float]:
    c_text = emit_c(program) + "\n" + timing_main(program, targets)
    exe = compile_c(c_text, workdir, tag, flags=(flag,), werror=False)
    result = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    return {name: float(ns) for name, ns in (line.split() for line in result.stdout.splitlines())}


def geomean(values) -> float:
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main() -> int:
    use_utf8_output()
    if find_gcc() is None:
        print("gcc is required for the timing measurements")
        return 2
    grids = load_grids()
    cost_rows, accuracy_rows = [], []
    with tempfile.TemporaryDirectory() as workdir:
        for bench in benchmark_names():
            compiled = compile_source(load_source(bench))
            grads = list(compiled.check.grads.values())
            if not grads:
                continue
            program, raw = compiled.program, compiled.raw
            points = {g.func: grid_points(grids[bench][g.func], program.functions[g.func].params)
                      for g in grads}
            targets = [(f, pts[0]) for f, pts in points.items()] + [(g.name, points[g.func][0]) for g in grads]

            times = {(variant, flag): measure(prog, targets, workdir, f"{bench}_{variant}_{flag[1:]}", flag)
                     for variant, prog in (("opt", program), ("raw", raw)) for flag in ("-O2", "-O0")}
            for g in grads:
                before, after = compiled.stats[g.name]
                primal = times["opt", "-O2"][g.func]
                cost_rows.append({
                    "benchmark": bench, "gradient": g.name,
                    "tac_before": before, "tac_after": after,
                    "primal_ns": primal, "grad_ns": times["opt", "-O2"][g.name],
                    "grad_over_primal": times["opt", "-O2"][g.name] / primal,
                    "raw_grad_ns_O2": times["raw", "-O2"][g.name],
                    "grad_ns_O0": times["opt", "-O0"][g.name],
                    "raw_grad_ns_O0": times["raw", "-O0"][g.name],
                })

            # Accuracy against closed forms, using the compiled optimised program.
            exact_grads = [g for g in grads if g.name in EXACT]
            if exact_grads:
                exe = compile_c(emit_c(program, harness=True), workdir, f"{bench}_harness")
                for g in exact_grads:
                    queries, steps = [], []
                    for point in points[g.func]:
                        hi, lo, h = step_points(point, g.index)
                        queries += [(g.name, point), (g.func, hi), (g.func, lo)]
                        steps.append((point, h))
                    answers = run_queries(exe, queries)
                    ad_err = fd_err = 0.0
                    for i, (point, h) in enumerate(steps):
                        (ad, _), (f_hi,), (f_lo,) = answers[3 * i: 3 * i + 3]
                        exact = EXACT[g.name](*point)
                        scale = abs(exact) or 1.0
                        ad_err = max(ad_err, abs(ad - exact) / scale)
                        fd_err = max(fd_err, abs((f_hi - f_lo) / (2 * h) - exact) / scale)
                    accuracy_rows.append({"benchmark": bench, "gradient": g.name, "points": len(steps),
                                          "ad_max_rel_error": ad_err, "fd_max_rel_error": fd_err})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("rq4_cost.csv", cost_rows), ("rq4_accuracy.csv", accuracy_rows)):
        with open(RESULTS_DIR / filename, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    cost_table = markdown_table(
        ["benchmark", "gradient", "TAC before", "TAC after", "function ns", "gradient ns",
         "gradient / function", "unoptimised TAC, -O2 ns", "-O0 ns", "unoptimised TAC, -O0 ns"],
        [[r["benchmark"], f"`{r['gradient']}`", r["tac_before"], r["tac_after"], f"{r['primal_ns']:.1f}",
          f"{r['grad_ns']:.1f}", f"{r['grad_over_primal']:.2f}", f"{r['raw_grad_ns_O2']:.1f}",
          f"{r['grad_ns_O0']:.1f}", f"{r['raw_grad_ns_O0']:.1f}"] for r in cost_rows])
    accuracy_table = markdown_table(
        ["benchmark", "gradient", "points", "generated gradient: largest relative error",
         "central difference: largest relative error"],
        [[r["benchmark"], f"`{r['gradient']}`", r["points"], f"{r['ad_max_rel_error']:.1e}",
          f"{r['fd_max_rel_error']:.1e}"] for r in accuracy_rows])

    before = sum(r["tac_before"] for r in cost_rows)
    after = sum(r["tac_after"] for r in cost_rows)
    summary = "\n".join([
        f"- {len(cost_rows)} gradients. Our optimiser removes {100 * (1 - after / before):.0f}% of the "
        f"three-address instructions ({before} to {after}).",
        f"- Gradient run time relative to the function (GCC -O2), geometric mean: "
        f"{geomean(r['grad_over_primal'] for r in cost_rows):.2f}x "
        f"(range {min(r['grad_over_primal'] for r in cost_rows):.2f}x to "
        f"{max(r['grad_over_primal'] for r in cost_rows):.2f}x). A central difference costs 2x "
        "and returns only the derivative; the gradient function also returns the value.",
        f"- Speed-up from our optimiser when GCC also optimises (-O2), geometric mean: "
        f"{geomean(r['raw_grad_ns_O2'] / r['grad_ns'] for r in cost_rows):.2f}x.",
        f"- Speed-up from our optimiser when GCC does not (-O0), geometric mean: "
        f"{geomean(r['raw_grad_ns_O0'] / r['grad_ns_O0'] for r in cost_rows):.2f}x.",
        f"- Against closed forms ({len(accuracy_rows)} gradients): largest relative error "
        f"{max(r['ad_max_rel_error'] for r in accuracy_rows):.1e} for generated gradients, "
        f"{max(r['fd_max_rel_error'] for r in accuracy_rows):.1e} for central differences.",
    ])
    note = ("Times are the best of five batches on one machine and vary from run to run; "
            "the instruction counts and accuracy figures are deterministic. Functions that take "
            "about a nanosecond are at the resolution of the timing loop, so their ratios are rough.")
    (RESULTS_DIR / "rq4_cost.md").write_text(
        f"# RQ4: cost of the generated gradients\n\n{summary}\n\n{note}\n\n{cost_table}\n\n"
        f"## Accuracy against closed forms\n\n{accuracy_table}\n", encoding="utf-8")
    print(cost_table)
    print()
    print(accuracy_table)
    print()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
