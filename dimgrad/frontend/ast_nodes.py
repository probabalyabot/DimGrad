"""Syntax tree node classes. Every node records the line and column it starts at."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

from ..units.units import Unit


@dataclass
class Node:
    line: int
    col: int


# --- unit expressions -------------------------------------------------------

@dataclass
class UnitExpr(Node):
    # (unit name, signed exponent) in source order; `1` contributes no factor.
    factors: list[tuple[str, Fraction]]
    placeholder: bool = False  # written as `?`


# --- expressions ------------------------------------------------------------

@dataclass
class Expr(Node):
    pass


@dataclass
class Num(Expr):
    value: float
    text: str
    unit: Optional[Unit] = field(default=None, compare=False)


@dataclass
class Name(Expr):
    name: str
    unit: Optional[Unit] = field(default=None, compare=False)


@dataclass
class BinOp(Expr):
    op: str  # + - * /
    lhs: Expr
    rhs: Expr
    unit: Optional[Unit] = field(default=None, compare=False)


@dataclass
class Neg(Expr):
    operand: Expr
    unit: Optional[Unit] = field(default=None, compare=False)


@dataclass
class Pow(Expr):
    base: Expr
    exp: int
    unit: Optional[Unit] = field(default=None, compare=False)


@dataclass
class Call(Expr):
    name: str
    args: list[Expr]
    unit: Optional[Unit] = field(default=None, compare=False)


@dataclass
class Cond(Node):
    op: str  # < > <= >= == !=
    lhs: Expr
    rhs: Expr


# --- statements -------------------------------------------------------------

@dataclass
class Block(Node):
    stmts: list


@dataclass
class Let(Node):
    name: str
    value: Expr


@dataclass
class Assign(Node):
    name: str
    value: Expr


@dataclass
class If(Node):
    cond: Cond
    then: Block
    els: Optional[Block]


@dataclass
class For(Node):
    var: str
    lo: int
    hi: int  # exclusive
    body: Block


@dataclass
class Return(Node):
    value: Expr


# --- declarations -----------------------------------------------------------

@dataclass
class UnitDecl(Node):
    names: list[tuple[str, int, int]]  # (name, line, col)


@dataclass
class UnitAlias(Node):
    name: str
    expr: UnitExpr


@dataclass
class ConstDecl(Node):
    name: str
    unit: UnitExpr
    value: float
    text: str


@dataclass
class Param(Node):
    name: str
    unit: UnitExpr


@dataclass
class FnDecl(Node):
    name: str
    params: list[Param]
    ret: UnitExpr
    body: Block


@dataclass
class GradDecl(Node):
    func: str
    wrt: str


@dataclass
class Program:
    decls: list


def iter_exprs(node):
    """Yield every expression under `node`, each parent before its children."""
    if isinstance(node, Block):
        for stmt in node.stmts:
            yield from iter_exprs(stmt)
    elif isinstance(node, (Let, Assign, Return)):
        yield from iter_exprs(node.value)
    elif isinstance(node, If):
        yield from iter_exprs(node.cond.lhs)
        yield from iter_exprs(node.cond.rhs)
        yield from iter_exprs(node.then)
        if node.els is not None:
            yield from iter_exprs(node.els)
    elif isinstance(node, For):
        yield from iter_exprs(node.body)
    elif isinstance(node, Expr):
        yield node
        if isinstance(node, BinOp):
            yield from iter_exprs(node.lhs)
            yield from iter_exprs(node.rhs)
        elif isinstance(node, Neg):
            yield from iter_exprs(node.operand)
        elif isinstance(node, Pow):
            yield from iter_exprs(node.base)
        elif isinstance(node, Call):
            for arg in node.args:
                yield from iter_exprs(arg)
