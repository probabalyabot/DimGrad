"""RQ3 (numerical methods): are correct Newton, gradient-descent and Euler steps
accepted, wrong ones rejected, and `?` units solved to the hand-derived result?

    python -m evaluation.methods
"""
from __future__ import annotations

import csv

from dimgrad import check_source
from dimgrad.glue.errors import DimGradError
from dimgrad.units.units import Unit

from .common import RESULTS_DIR, markdown_table, use_utf8_output

kg, m, s, K = (Unit.of(n) for n in ("kg", "m", "s", "K"))
J = kg * m ** 2 / s ** 2

PRELUDE = """
unit kg, m, s, K;
unit J = kg*m^2/s^2;
const stiffness: J/m^2 = 50.0;
const speed: m/s = 3.0;

fn g(x: m, t: s) -> m^2 { return x^2 - (speed * t)^2; }
grad g wrt x;
grad g wrt t;

fn f(x: m) -> J { return 0.5 * stiffness * x^2; }
grad f wrt x;

fn rate(y: K, r: 1/s) -> K/s { return -r * y; }
"""

# (method, description, the step function, expected outcome, expected unit of the `?` parameter)
CASES = [
    ("Newton", "x - g/g'  (derivative with respect to x)",
     "fn step(x: m, t: s) -> m { return x - g(x, t) / g_grad_x(x, t); }", "accept", None),
    ("Newton", "derivative with respect to the wrong variable",
     "fn step(x: m, t: s) -> m { return x - g(x, t) / g_grad_t(x, t); }", "reject", None),
    ("Newton", "division left out: x - g",
     "fn step(x: m, t: s) -> m { return x - g(x, t); }", "reject", None),
    ("Newton", "quotient inverted: x - g'/g",
     "fn step(x: m, t: s) -> m { return x - g_grad_x(x, t) / g(x, t); }", "reject", None),

    ("Gradient descent", "learning rate declared with ?",
     "fn step(x: m, eta: ?) -> m { return x - eta * f_grad_x(x); }", "accept", m ** 2 / J),
    ("Gradient descent", "learning rate declared as m^2/J",
     "fn step(x: m, eta: m^2/J) -> m { return x - eta * f_grad_x(x); }", "accept", None),
    ("Gradient descent", "learning rate declared unitless",
     "fn step(x: m, eta: 1) -> m { return x - eta * f_grad_x(x); }", "reject", None),
    ("Gradient descent", "learning rate declared as m/J",
     "fn step(x: m, eta: m/J) -> m { return x - eta * f_grad_x(x); }", "reject", None),
    ("Gradient descent", "function value used instead of its gradient",
     "fn step(x: m, eta: m^2/J) -> m { return x - eta * f(x); }", "reject", None),
    ("Gradient descent", "learning rate left out: x - f'",
     "fn step(x: m) -> m { return x - f_grad_x(x); }", "reject", None),

    ("Euler", "step size declared with ?",
     "fn step(y: K, r: 1/s, h: ?) -> K { return y + h * rate(y, r); }", "accept", s),
    ("Euler", "step size declared as s",
     "fn step(y: K, r: 1/s, h: s) -> K { return y + h * rate(y, r); }", "accept", None),
    ("Euler", "step size declared unitless",
     "fn step(y: K, r: 1/s, h: 1) -> K { return y + h * rate(y, r); }", "reject", None),
    ("Euler", "step size left out: y + f(y)",
     "fn step(y: K, r: 1/s) -> K { return y + rate(y, r); }", "reject", None),
    ("Euler", "state used instead of its rate of change: y + h*y",
     "fn step(y: K, r: 1/s, h: s) -> K { return y + h * y; }", "reject", None),
]


def run_case(step: str, expected_unit):
    """(outcome, detail) for one step function."""
    try:
        result = check_source(PRELUDE + step)
    except DimGradError as exc:
        return "reject", exc.diagnostics[0].message
    func = result.funcs["step"]
    if func.inferred:
        solved = func.param_units[min(func.inferred)]
        detail = f"solved: {result.fmt(solved)}"
        if expected_unit is not None:
            detail += " (matches the hand result)" if solved == expected_unit else " (DIFFERS from the hand result)"
            if solved != expected_unit:
                return "wrong unit", detail
        return "accept", detail
    return "accept", ""


def main() -> int:
    use_utf8_output()
    rows = []
    for method, description, step, expected, expected_unit in CASES:
        outcome, detail = run_case(step, expected_unit)
        rows.append({"method": method, "case": description, "expected": expected, "outcome": outcome,
                     "as expected": "yes" if outcome == expected else "NO", "detail": detail})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rq3_numerical_methods.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = markdown_table(list(rows[0]), [list(r.values()) for r in rows])
    wrong = sum(r["as expected"] != "yes" for r in rows)
    summary = f"{len(rows) - wrong} of {len(rows)} cases behave as derived by hand."
    (RESULTS_DIR / "rq3_numerical_methods.md").write_text(
        f"# RQ3: units of numerical-method updates\n\n{summary}\n\n{table}\n", encoding="utf-8")
    print(table)
    print()
    print(summary)
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
