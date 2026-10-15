"""Symbol table: nested variable scopes plus the records kept for global names."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..frontend import ast_nodes as A
from .units import Unit


@dataclass
class VarInfo:
    name: str
    unit: Unit
    kind: str  # "param", "let", "loop" or "const"
    line: int


@dataclass
class ConstInfo:
    name: str
    unit: Unit
    value: float
    line: int


@dataclass
class FuncInfo:
    name: str
    node: A.FnDecl
    param_names: list[str]
    param_units: list[Unit]
    ret_unit: Unit
    inferred: set[int] = field(default_factory=set)  # indices of `?` parameters
    calls: set[str] = field(default_factory=set)     # user functions this one depends on
    uses_grad: bool = False                           # calls a gradient function, directly or not


@dataclass
class GradInfo:
    name: str  # the callable, e.g. energy_grad_v
    func: str
    wrt: str
    index: int  # position of `wrt` among the parameters
    node: A.GradDecl
    unit: Optional[Unit] = None  # [func] / [wrt], known once func is checked


class Scopes:
    """A stack of local scopes. Shadowing is not allowed, so lookups never hide a name."""

    def __init__(self):
        self.stack: list[dict[str, VarInfo]] = []

    def push(self):
        self.stack.append({})

    def pop(self):
        self.stack.pop()

    def lookup(self, name: str) -> Optional[VarInfo]:
        for scope in reversed(self.stack):
            if name in scope:
                return scope[name]
        return None

    def declare(self, info: VarInfo):
        self.stack[-1][info.name] = info
