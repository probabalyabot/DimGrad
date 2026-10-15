"""Optimiser for three-address code.

A forward pass does constant folding, constant and copy propagation, the
algebraic identities (x*1, x+0, 0*x, ...) and common-subexpression
elimination. A backward pass removes dead code. The two repeat until nothing
changes. Named variables can be reassigned, so facts about a name are dropped
when it is written, merged at the end of an if/else, and iterated to a fixed
point around a for loop.

Note: 0*x is folded to 0, which differs from IEEE 754 when x is inf or nan.
Named constants are never folded, so every non-zero literal stays unitless
and the unit verifier can still check optimised code.
"""
from __future__ import annotations

import dataclasses
import math
from collections import Counter

from ..middle import tac
from ..middle.interp import _RELOPS, apply

_COMMUTATIVE = ("add", "mul")


class _Env:
    """Facts that hold at a program point."""

    def __init__(self, val=None, avail=None):
        self.val: dict = dict(val or {})      # name -> constant, or the name it is a copy of
        self.avail: dict = dict(avail or {})  # (op, args) -> name already holding that value

    def copy(self) -> "_Env":
        return _Env(self.val, self.avail)

    def kill(self, name: str):
        self.val.pop(name, None)
        for key in [k for k, v in self.val.items() if v == name]:
            del self.val[key]
        for key in [k for k, holder in self.avail.items() if holder == name or name in k[1]]:
            del self.avail[key]

    def meet(self, other: "_Env") -> "_Env":
        val = {k: v for k, v in self.val.items() if k in other.val and other.val[k] == v}
        avail = {k: v for k, v in self.avail.items() if other.avail.get(k) == v}
        return _Env(val, avail)

    def replace_with(self, other: "_Env"):
        self.val, self.avail = other.val, other.avail

    def __eq__(self, other) -> bool:
        return self.val == other.val and self.avail == other.avail


def _is(operand, value: float) -> bool:
    return isinstance(operand, float) and operand == value


def _fold(op: str, args: tuple):
    """Simplify one operation; the result is either the same kind of operation or a copy."""
    if op == "call":
        return op, args
    values = args[:1] if op == "pow" else args
    if all(isinstance(v, float) for v in values):
        try:
            result = apply(op, list(args))
        except (ArithmeticError, ValueError):
            result = math.nan
        if math.isfinite(result):
            return "copy", (result,)
    a = args[0]
    b = args[1] if len(args) > 1 else None
    if op == "add":
        if _is(a, 0.0):
            return "copy", (b,)
        if _is(b, 0.0):
            return "copy", (a,)
    elif op == "sub":
        if _is(b, 0.0):
            return "copy", (a,)
        if _is(a, 0.0):
            return "neg", (b,)
        if isinstance(a, str) and a == b:
            return "copy", (0.0,)
    elif op == "mul":
        if _is(a, 0.0) or _is(b, 0.0):
            return "copy", (0.0,)
        if _is(a, 1.0):
            return "copy", (b,)
        if _is(b, 1.0):
            return "copy", (a,)
    elif op == "div":
        if _is(a, 0.0):
            return "copy", (0.0,)
        if _is(b, 1.0):
            return "copy", (a,)
    elif op == "pow":
        if b == 0:
            return "copy", (1.0,)
        if b == 1:
            return "copy", (a,)
    return op, args


def _key(op: str, args: tuple):
    if op in _COMMUTATIVE:
        args = tuple(sorted(args, key=repr))
    return op, args


def _simplify(instr: tac.Instr, env: _Env) -> tac.Instr:
    def known(operand):
        return env.val.get(operand, operand) if isinstance(operand, str) else operand

    op, args = instr.op, tac.with_operands(instr, known)
    key = None
    if op != "copy":
        op, args = _fold(op, args)
    if op != "copy":
        key = _key(op, args)
        if key in env.avail:
            op, args = "copy", (env.avail[key],)
    env.kill(instr.dst)
    if op == "copy":
        if args[0] != instr.dst:
            env.val[instr.dst] = args[0]
    elif instr.dst not in args:
        env.avail[key] = instr.dst
    return tac.Instr(instr.dst, op, args, instr.unit, instr.line)


def _assigned(body: list) -> set[str]:
    names = set()
    for item in tac.walk(body):
        if isinstance(item, tac.Instr):
            names.add(item.dst)
        elif isinstance(item, tac.For):
            names.add(item.var)
    return names


def _forward(body: list, env: _Env) -> list:
    def known(operand):
        return env.val.get(operand, operand) if isinstance(operand, str) else operand

    out = []
    for item in body:
        if isinstance(item, tac.Instr):
            out.append(_simplify(item, env))
        elif isinstance(item, tac.If):
            lhs, rhs = known(item.lhs), known(item.rhs)
            if isinstance(lhs, float) and isinstance(rhs, float):
                out.extend(_forward(item.then if _RELOPS[item.relop](lhs, rhs) else item.els, env))
                continue
            then_env, else_env = env.copy(), env.copy()
            then = _forward(item.then, then_env)
            els = _forward(item.els, else_env)
            env.replace_with(then_env.meet(else_env))
            out.append(tac.If(lhs, item.relop, rhs, then, els, item.line))
        elif isinstance(item, tac.For):
            if item.lo >= item.hi:
                continue
            entry = env.copy()
            entry.kill(item.var)
            # Find facts that hold at the top of every iteration: start from
            # what holds on entry and drop whatever one trip round the body breaks.
            head = entry.copy()
            for _ in range(8):
                after = head.copy()
                _forward(item.body, after)
                narrowed = entry.meet(after)
                if narrowed == head:
                    break
                head = narrowed
            else:
                head = entry.copy()
                for name in _assigned(item.body):
                    head.kill(name)
            new_body = _forward(item.body, head.copy())
            env.replace_with(head)
            out.append(tac.For(item.var, item.lo, item.hi, new_body, item.line))
        elif isinstance(item, tac.Return):
            tangent = None if item.tangent is None else known(item.tangent)
            out.append(tac.Return(known(item.value), tangent, item.line))
    return out


def _names(*operands) -> set[str]:
    return {o for o in operands if isinstance(o, str)}


def _dce(body: list, live_out: set[str]):
    """Remove assignments whose value is never used. Returns (new body, live-in set)."""
    live = set(live_out)
    out = []
    for item in reversed(body):
        if isinstance(item, tac.Instr):
            if item.dst not in live:
                continue
            if item.op == "copy" and item.args[0] == item.dst:
                continue
            live.discard(item.dst)
            live |= _names(*tac.operands(item))
            out.append(item)
        elif isinstance(item, tac.If):
            then, live_then = _dce(item.then, live)
            els, live_else = _dce(item.els, live)
            if not then and not els:
                continue
            live = live_then | live_else | _names(item.lhs, item.rhs)
            out.append(tac.If(item.lhs, item.relop, item.rhs, then, els, item.line))
        elif isinstance(item, tac.For):
            # What is live at the end of the body is what is live after the
            # loop plus what the next iteration reads.
            loop_live = set(live)
            while True:
                _, live_in = _dce(item.body, loop_live)
                if live_in <= loop_live:
                    break
                loop_live |= live_in
            new_body, _ = _dce(item.body, loop_live)
            if not new_body:
                continue
            live = loop_live - {item.var}
            out.append(tac.For(item.var, item.lo, item.hi, new_body, item.line))
        elif isinstance(item, tac.Return):
            live = _names(item.value, item.tangent)
            out.append(item)
    out.reverse()
    return out, live


def _use_counts(body: list) -> Counter:
    uses: Counter = Counter()
    for item in tac.walk(body):
        if isinstance(item, tac.Instr):
            uses.update(o for o in tac.operands(item) if isinstance(o, str))
        elif isinstance(item, tac.If):
            uses.update(o for o in (item.lhs, item.rhs) if isinstance(o, str))
        elif isinstance(item, tac.Return):
            uses.update(o for o in (item.value, item.tangent) if isinstance(o, str))
    return uses


def _coalesce(body: list, variables: set[str], uses: Counter) -> list:
    """Turn `t = a op b; x = t` into `x = a op b` when t has no other use."""
    out = []
    for item in body:
        if isinstance(item, tac.Instr):
            prev = out[-1] if out else None
            src = item.args[0]
            if (item.op == "copy" and isinstance(prev, tac.Instr) and isinstance(src, str)
                    and prev.dst == src and src not in variables and uses[src] == 1):
                out[-1] = tac.Instr(item.dst, prev.op, prev.args, item.unit, prev.line)
            else:
                out.append(item)
        elif isinstance(item, tac.If):
            out.append(tac.If(item.lhs, item.relop, item.rhs, _coalesce(item.then, variables, uses),
                              _coalesce(item.els, variables, uses), item.line))
        elif isinstance(item, tac.For):
            out.append(tac.For(item.var, item.lo, item.hi, _coalesce(item.body, variables, uses), item.line))
        else:
            out.append(item)
    return out


def optimize_function(func: tac.Function) -> tac.Function:
    body = func.body
    for _ in range(50):
        new_body, _ = _dce(_forward(body, _Env()), set())
        if new_body == body:
            break
        body = new_body
    body = _coalesce(body, func.vars, _use_counts(body))
    kept = _assigned(body) | set(func.params)
    units = {name: unit for name, unit in func.units.items() if name in kept}
    return dataclasses.replace(func, body=body, units=units, vars=func.vars & kept)
