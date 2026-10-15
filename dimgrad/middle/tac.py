"""Three-address code. Every instruction carries the unit of the value it defines.

An operand is a name (str) or a numeric constant (float). `if` and `for` stay
as structured blocks so the C emitter can reproduce them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..units.units import Unit

BUILTIN_OPS = ("sqrt", "cbrt", "sin", "cos", "exp", "log", "normcdf")
BINARY_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


@dataclass
class Instr:
    """dst = op(args).

    op is copy, neg, add, sub, mul, div, pow, call or a built-in name.
    For pow the arguments are (base, integer exponent); for call they are
    (function name, operand, ...).
    """
    dst: str
    op: str
    args: tuple
    unit: Unit
    line: int = 0


@dataclass
class If:
    lhs: object
    relop: str
    rhs: object
    then: list
    els: list
    line: int = 0


@dataclass
class For:
    var: str
    lo: int
    hi: int  # exclusive
    body: list
    line: int = 0


@dataclass
class Return:
    value: object
    tangent: object = None  # the derivative, in a gradient function
    line: int = 0


@dataclass
class Function:
    name: str
    params: list[str]
    ret_unit: Unit
    body: list
    units: dict[str, Unit]  # unit of every name the function defines, parameters included
    vars: set[str]          # names that may be assigned more than once
    kind: str = "primal"    # or "grad"
    of: Optional[str] = None            # grad: the differentiated function
    wrt: Optional[str] = None           # grad: the differentiation variable
    value_unit: Optional[Unit] = None   # grad: unit of the primal value
    tangents: dict[str, str] = field(default_factory=dict)  # grad: name -> its tangent's name
    line: int = 0


@dataclass
class Program:
    consts: dict[str, tuple[Unit, float]]
    functions: dict[str, Function]
    unit_order: list[str]
    aliases: dict[str, Unit]

    def fmt(self, unit: Unit) -> str:
        return unit.format(self.unit_order)


class NameGen:
    """Hands out names that do not clash with any name already in use."""

    def __init__(self, used):
        self.used = set(used)
        self.counters: dict[str, int] = {}

    def temp(self, prefix: str = "t") -> str:
        n = self.counters.get(prefix, 0)
        while True:
            n += 1
            name = f"{prefix}{n}"
            if name not in self.used:
                break
        self.counters[prefix] = n
        self.used.add(name)
        return name

    def named(self, base: str) -> str:
        name, n = base, 1
        while name in self.used:
            n += 1
            name = f"{base}_{n}"
        self.used.add(name)
        return name


def operands(instr: Instr) -> tuple:
    """The value operands of an instruction (not a callee name or a pow exponent)."""
    if instr.op == "call":
        return instr.args[1:]
    if instr.op == "pow":
        return instr.args[:1]
    return instr.args


def with_operands(instr: Instr, fn) -> tuple:
    """The instruction's args with `fn` applied to each value operand."""
    if instr.op == "call":
        return (instr.args[0],) + tuple(fn(a) for a in instr.args[1:])
    if instr.op == "pow":
        return (fn(instr.args[0]), instr.args[1])
    return tuple(fn(a) for a in instr.args)


def walk(body: list):
    """Every item in a body, nested blocks included."""
    for item in body:
        yield item
        if isinstance(item, If):
            yield from walk(item.then)
            yield from walk(item.els)
        elif isinstance(item, For):
            yield from walk(item.body)


def count_instrs(body: list) -> int:
    return sum(1 for item in walk(body) if isinstance(item, Instr))


def all_names(func: Function, program: Program) -> set[str]:
    return set(func.units) | set(func.params) | set(program.consts)


# --- printing ---------------------------------------------------------------

def operand_text(operand) -> str:
    return repr(operand) if isinstance(operand, float) else str(operand)


def instr_text(instr: Instr) -> str:
    a = [operand_text(x) for x in operands(instr)]
    if instr.op == "copy":
        rhs = a[0]
    elif instr.op == "neg":
        rhs = f"-{a[0]}"
    elif instr.op in BINARY_SYMBOLS:
        rhs = f"{a[0]} {BINARY_SYMBOLS[instr.op]} {a[1]}"
    elif instr.op == "pow":
        rhs = f"{a[0]} ^ {instr.args[1]}"
    elif instr.op == "call":
        rhs = f"{instr.args[0]}({', '.join(a)})"
    else:
        rhs = f"{instr.op}({a[0]})"
    return f"{instr.dst} = {rhs}"


def _format_body(body: list, depth: int, fmt, out: list[str]):
    pad = "    " * depth
    for item in body:
        if isinstance(item, Instr):
            out.append(f"{pad + instr_text(item):<44}  [{fmt(item.unit)}]")
        elif isinstance(item, If):
            out.append(f"{pad}if {operand_text(item.lhs)} {item.relop} {operand_text(item.rhs)}:")
            _format_body(item.then, depth + 1, fmt, out)
            if item.els:
                out.append(f"{pad}else:")
                _format_body(item.els, depth + 1, fmt, out)
        elif isinstance(item, For):
            out.append(f"{pad}for {item.var} in {item.lo}..{item.hi}:")
            _format_body(item.body, depth + 1, fmt, out)
        elif isinstance(item, Return):
            text = f"{pad}return {operand_text(item.value)}"
            if item.tangent is not None:
                text += f"    derivative: {operand_text(item.tangent)}"
            out.append(text)


def format_function(func: Function, program: Program) -> str:
    fmt = program.fmt
    params = ", ".join(f"{p}: {fmt(func.units[p])}" for p in func.params)
    out = [f"{func.name}({params}) -> {fmt(func.ret_unit)}"]
    _format_body(func.body, 1, fmt, out)
    return "\n".join(out)


def format_program(program: Program) -> str:
    return "\n\n".join(format_function(f, program) for f in program.functions.values())
