"""The pipeline: source text -> checked tree -> three-address code -> gradients -> optimised code."""
from __future__ import annotations

from dataclasses import dataclass

from ..backend.optimizer import optimize_function
from ..diff.autodiff import differentiate
from ..diff.inline import inline_calls
from ..diff.verify import verify_function
from ..frontend.parser import parse
from ..middle import tac
from ..middle.lower import lower_program
from ..units.checker import CheckResult, check


@dataclass
class Compiled:
    check: CheckResult
    raw: tac.Program      # functions and gradients as generated, before optimisation
    program: tac.Program  # what gets emitted
    stats: dict[str, tuple[int, int]]  # function name -> (instructions before, after)


def check_source(source: str) -> CheckResult:
    return check(parse(source))


def compile_source(source: str, optimize: bool = True, verify: bool = True) -> Compiled:
    checked = check_source(source)
    raw = lower_program(checked)
    for grad in checked.grads.values():
        inlined = inline_calls(raw.functions[grad.func], raw)
        raw.functions[grad.name] = differentiate(inlined, grad.wrt, raw, name=grad.name)
    if verify:
        for func in raw.functions.values():
            verify_function(func, raw)

    program = tac.Program(raw.consts, dict(raw.functions), raw.unit_order, raw.aliases)
    stats = {}
    for name, func in raw.functions.items():
        if optimize:
            program.functions[name] = optimize_function(func)
        stats[name] = (tac.count_instrs(func.body), tac.count_instrs(program.functions[name].body))
    if verify and optimize:
        for func in program.functions.values():
            verify_function(func, program)
    return Compiled(checked, raw, program, stats)
