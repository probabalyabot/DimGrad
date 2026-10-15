"""Random programs: the optimiser must not change results, and gradients must match finite differences."""
import math
import random

import pytest

from dimgrad import compile_source
from dimgrad.glue.fd import central_difference, relative_error
from dimgrad.middle.interp import evaluate

LEAVES = ["0", "1", "2", "0.5", "3"]


def gen_expr(rng, names, depth):
    if depth == 0 or rng.random() < 0.25:
        return rng.choice(names + LEAVES if rng.random() < 0.3 else names)
    a = gen_expr(rng, names, depth - 1)
    b = gen_expr(rng, names, depth - 1)
    kind = rng.choice(["+", "-", "*", "*", "/", "neg", "pow", "sin", "cos", "exp", "sqrt", "cbrt", "log"])
    if kind in "+-*":
        return f"({a} {kind} {b})"
    if kind == "/":
        return f"({a} / (({b})^2 + 1))"
    if kind == "neg":
        return f"(-{a})"
    if kind == "pow":
        return f"(1 + ({a})^2)^{rng.choice([0, 1, 2, 3, -1, -2])}"
    if kind == "exp":
        return f"exp(sin({a}))"
    if kind == "sqrt":
        return f"sqrt(({a})^2 + 1)"
    if kind == "cbrt":
        return f"cbrt(({a})^2 + 1)"
    if kind == "log":
        return f"log(({a})^2 + 1)"
    return f"{kind}({a})"


def gen_block(rng, readable, writable, depth, counter, pad):
    lines = []
    readable, writable = list(readable), list(writable)
    for _ in range(rng.randint(1, 4)):
        choice = rng.random()
        if choice < 0.4 or not writable:
            counter[0] += 1
            name = f"v{counter[0]}"
            lines.append(f"{pad}let {name} = {gen_expr(rng, readable, 2)};")
            readable.append(name)
            writable.append(name)
        elif choice < 0.65:
            lines.append(f"{pad}{rng.choice(writable)} = {gen_expr(rng, readable, 2)};")
        elif choice < 0.85 and depth > 0:
            lines.append(f"{pad}if ({gen_expr(rng, readable, 1)} < {gen_expr(rng, readable, 1)}) {{")
            lines += gen_block(rng, readable, writable, depth - 1, counter, pad + "    ")
            lines.append(f"{pad}}} else {{")
            lines += gen_block(rng, readable, writable, depth - 1, counter, pad + "    ")
            lines.append(f"{pad}}}")
        elif depth > 0:
            counter[0] += 1
            var = f"i{counter[0]}"
            lines.append(f"{pad}for {var} in 0..{rng.randint(0, 3)} {{")
            lines += gen_block(rng, readable + [var], writable, depth - 1, counter, pad + "    ")
            lines.append(f"{pad}}}")
    return lines


def gen_program(seed):
    rng = random.Random(seed)
    counter = [0]
    lines = ["fn helper(p: 1, q: 1) -> 1 {",
             f"    if (p < q) {{ return {gen_expr(rng, ['p', 'q'], 2)}; }} else {{ return {gen_expr(rng, ['p', 'q'], 2)}; }}",
             "}",
             "fn f(x: 1, y: 1) -> 1 {",
             "    let acc = helper(x, y);"]
    lines += gen_block(rng, ["x", "y", "acc"], ["acc"], 2, counter, "    ")
    names = ["x", "y", "acc"]
    lines.append(f"    return acc + {gen_expr(rng, names, 2)} * helper(y, acc);")
    lines += ["}", "grad f wrt x;", "grad f wrt y;"]
    return "\n".join(lines)


def same(a, b):
    return a == b or abs(a - b) <= 1e-9 * max(abs(a), abs(b))


@pytest.mark.parametrize("seed", range(300))
def test_random_program(seed):
    source = gen_program(seed)
    compiled = compile_source(source)  # runs the unit verifier before and after optimisation
    for point in ([0.3, -0.7], [1.9, 0.4], [-1.2, 2.3]):
        for name in ("f", "f_grad_x", "f_grad_y"):
            raw = evaluate(compiled.raw, name, point)
            opt = evaluate(compiled.program, name, point)
            raw, opt = (raw, opt) if isinstance(raw, tuple) else ((raw,), (opt,))
            if all(math.isfinite(v) for v in raw):
                assert all(same(a, b) for a, b in zip(raw, opt)), (name, point, raw, opt, source)
        value = evaluate(compiled.program, "f", point)
        if not math.isfinite(value) or abs(value) > 1e6:
            continue
        for index, name in enumerate(("f_grad_x", "f_grad_y")):
            ad, _ = evaluate(compiled.program, name, point)
            fd, h = central_difference(lambda args: evaluate(compiled.program, "f", args), point, index)
            # Loose: a random program may sit next to a branch flip or be badly conditioned.
            if math.isfinite(ad) and math.isfinite(fd) and abs(ad) < 1e6:
                assert relative_error(ad, fd, value, h, tol=1e-3) < 1e-3, (name, point, ad, fd, source)
