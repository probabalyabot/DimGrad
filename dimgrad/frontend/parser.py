"""Recursive-descent parser. One method per grammar rule (see docs/grammar.md)."""
from __future__ import annotations

from fractions import Fraction

from ..glue.errors import Diagnostic, DimGradError
from . import ast_nodes as A
from .lexer import Token, tokenize

RELOPS = ("<", ">", "<=", ">=", "==", "!=")


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # --- token helpers ------------------------------------------------------

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.kind != "EOF":
            self.pos += 1
        return tok

    def accept(self, kind: str):
        if self.peek().kind == kind:
            return self.advance()
        return None

    def error(self, tok: Token, expected: str):
        found = "end of file" if tok.kind == "EOF" else f"'{tok.text}'"
        raise DimGradError(Diagnostic(tok.line, tok.col, f"expected {expected} but found {found}"))

    def expect(self, kind: str, what: str | None = None) -> Token:
        if self.peek().kind != kind:
            self.error(self.peek(), what or f"'{kind}'")
        return self.advance()

    def integer(self, what: str) -> int:
        negative = self.accept("-") is not None
        value = int(self.expect("INT", what).text)
        return -value if negative else value

    # --- declarations -------------------------------------------------------

    def program(self) -> A.Program:
        decls = []
        while self.peek().kind != "EOF":
            kind = self.peek().kind
            if kind == "unit":
                decls.append(self.unit_decl())
            elif kind == "const":
                decls.append(self.const_decl())
            elif kind == "fn":
                decls.append(self.function())
            elif kind == "grad":
                decls.append(self.grad_decl())
            else:
                self.error(self.peek(), "a declaration (unit, const, fn or grad)")
        return A.Program(decls)

    def unit_decl(self):
        kw = self.expect("unit")
        first = self.expect("NAME", "a unit name")
        if self.accept("="):
            expr = self.unit_expr()
            self.expect(";")
            return A.UnitAlias(kw.line, kw.col, first.text, expr)
        names = [(first.text, first.line, first.col)]
        while self.accept(","):
            tok = self.expect("NAME", "a unit name")
            names.append((tok.text, tok.line, tok.col))
        self.expect(";")
        return A.UnitDecl(kw.line, kw.col, names)

    def const_decl(self):
        kw = self.expect("const")
        name = self.expect("NAME", "a constant name")
        self.expect(":")
        unit = self.unit_expr()
        self.expect("=")
        negative = self.accept("-") is not None
        tok = self.peek()
        if tok.kind not in ("INT", "NUMBER"):
            self.error(tok, "a number")
        self.advance()
        self.expect(";")
        text = ("-" if negative else "") + tok.text
        return A.ConstDecl(kw.line, kw.col, name.text, unit, float(text), text)

    def function(self):
        kw = self.expect("fn")
        name = self.expect("NAME", "a function name")
        self.expect("(")
        params = []
        if self.peek().kind != ")":
            params.append(self.param())
            while self.accept(","):
                params.append(self.param())
        self.expect(")")
        self.expect("->")
        ret = self.unit_expr()
        body = self.block()
        return A.FnDecl(kw.line, kw.col, name.text, params, ret, body)

    def param(self):
        name = self.expect("NAME", "a parameter name")
        self.expect(":")
        unit = self.unit_expr(allow_placeholder=True)
        return A.Param(name.line, name.col, name.text, unit)

    def grad_decl(self):
        kw = self.expect("grad")
        func = self.expect("NAME", "a function name")
        self.expect("wrt")
        wrt = self.expect("NAME", "a parameter name")
        self.expect(";")
        return A.GradDecl(kw.line, kw.col, func.text, wrt.text)

    # --- unit expressions ---------------------------------------------------

    def unit_expr(self, allow_placeholder: bool = False) -> A.UnitExpr:
        start = self.peek()
        if start.kind == "?":
            if not allow_placeholder:
                raise DimGradError(Diagnostic(
                    start.line, start.col, "'?' is only allowed as the unit of a parameter"))
            self.advance()
            return A.UnitExpr(start.line, start.col, [], placeholder=True)
        factors: list = []
        self.unit_term(factors, 1)
        while self.peek().kind in ("*", "/"):
            sign = 1 if self.advance().kind == "*" else -1
            self.unit_term(factors, sign)
        return A.UnitExpr(start.line, start.col, factors)

    def unit_term(self, factors: list, sign: int):
        tok = self.peek()
        if tok.kind == "INT" and tok.text == "1":
            self.advance()
            return
        name = self.expect("NAME", "a unit name")
        exponent = Fraction(1)
        if self.accept("^"):
            exponent = self.unit_exponent()
        factors.append((name.text, sign * exponent))

    def unit_exponent(self) -> Fraction:
        if self.accept("("):
            num = self.integer("an integer")
            self.expect("/")
            den_tok = self.peek()
            den = int(self.expect("INT", "an integer").text)
            if den == 0:
                raise DimGradError(Diagnostic(den_tok.line, den_tok.col, "exponent has a zero denominator"))
            self.expect(")")
            return Fraction(num, den)
        return Fraction(self.integer("an integer exponent"))

    # --- statements ---------------------------------------------------------

    def block(self) -> A.Block:
        brace = self.expect("{")
        stmts = []
        while self.peek().kind != "}":
            if self.peek().kind == "EOF":
                self.error(self.peek(), "'}'")
            stmts.append(self.statement())
        self.expect("}")
        return A.Block(brace.line, brace.col, stmts)

    def statement(self):
        tok = self.peek()
        if tok.kind == "let":
            self.advance()
            name = self.expect("NAME", "a variable name")
            self.expect("=")
            value = self.expr()
            self.expect(";")
            return A.Let(tok.line, tok.col, name.text, value)
        if tok.kind == "NAME":
            self.advance()
            self.expect("=")
            value = self.expr()
            self.expect(";")
            return A.Assign(tok.line, tok.col, tok.text, value)
        if tok.kind == "if":
            self.advance()
            self.expect("(")
            cond = self.cond()
            self.expect(")")
            then = self.block()
            els = self.block() if self.accept("else") else None
            return A.If(tok.line, tok.col, cond, then, els)
        if tok.kind == "for":
            self.advance()
            var = self.expect("NAME", "a loop variable name")
            self.expect("in")
            lo = self.integer("an integer loop bound")
            self.expect("..")
            hi = self.integer("an integer loop bound")
            body = self.block()
            return A.For(tok.line, tok.col, var.text, lo, hi, body)
        if tok.kind == "return":
            self.advance()
            value = self.expr()
            self.expect(";")
            return A.Return(tok.line, tok.col, value)
        self.error(tok, "a statement")

    def cond(self) -> A.Cond:
        lhs = self.expr()
        tok = self.peek()
        if tok.kind not in RELOPS:
            self.error(tok, "a comparison operator")
        self.advance()
        rhs = self.expr()
        return A.Cond(tok.line, tok.col, tok.kind, lhs, rhs)

    # --- expressions --------------------------------------------------------

    def expr(self) -> A.Expr:
        node = self.term()
        while self.peek().kind in ("+", "-"):
            tok = self.advance()
            node = A.BinOp(tok.line, tok.col, tok.kind, node, self.term())
        return node

    def term(self) -> A.Expr:
        node = self.unary()
        while self.peek().kind in ("*", "/"):
            tok = self.advance()
            node = A.BinOp(tok.line, tok.col, tok.kind, node, self.unary())
        return node

    def unary(self) -> A.Expr:
        tok = self.accept("-")
        if tok:
            return A.Neg(tok.line, tok.col, self.unary())
        return self.power()

    def power(self) -> A.Expr:
        base = self.atom()
        tok = self.accept("^")
        if tok:
            return A.Pow(tok.line, tok.col, base, self.integer("an integer exponent"))
        return base

    def atom(self) -> A.Expr:
        tok = self.peek()
        if tok.kind in ("INT", "NUMBER"):
            self.advance()
            return A.Num(tok.line, tok.col, float(tok.text), tok.text)
        if tok.kind == "NAME":
            self.advance()
            if not self.accept("("):
                return A.Name(tok.line, tok.col, tok.text)
            args = []
            if self.peek().kind != ")":
                args.append(self.expr())
                while self.accept(","):
                    args.append(self.expr())
            self.expect(")")
            return A.Call(tok.line, tok.col, tok.text, args)
        if tok.kind == "(":
            self.advance()
            node = self.expr()
            self.expect(")")
            return node
        self.error(tok, "an expression")


def parse(source: str) -> A.Program:
    return Parser(tokenize(source)).program()
