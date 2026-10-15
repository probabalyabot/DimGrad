"""Scanner: source text to tokens with a kind, text, line and column."""
from __future__ import annotations

from dataclasses import dataclass

from ..glue.errors import Diagnostic, DimGradError

KEYWORDS = {"unit", "const", "fn", "let", "if", "else", "for", "in", "return", "grad", "wrt"}

# Longest first, so that "->" is not read as "-" followed by ">".
PUNCTUATION = [
    "->", "..", "<=", ">=", "==", "!=",
    "(", ")", "{", "}", ",", ";", ":", "+", "-", "*", "/", "^", "<", ">", "=", "?",
]


@dataclass(frozen=True)
class Token:
    kind: str  # NAME, INT, NUMBER, EOF, or the keyword / punctuation text itself
    text: str
    line: int
    col: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i, line, col = 0, 1, 1
    n = len(source)
    while i < n:
        ch = source[i]
        if ch == "\n":
            i, line, col = i + 1, line + 1, 1
            continue
        if ch in " \t\r":
            i, col = i + 1, col + 1
            continue
        if source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
            continue
        start = i
        if ch.isdigit():
            kind = "INT"
            while i < n and source[i].isdigit():
                i += 1
            # A '.' only starts a fraction when a digit follows, so `0..10` stays a range.
            if i + 1 < n and source[i] == "." and source[i + 1].isdigit():
                kind = "NUMBER"
                i += 1
                while i < n and source[i].isdigit():
                    i += 1
            if i < n and source[i] in "eE":
                j = i + 1
                if j < n and source[j] in "+-":
                    j += 1
                if j < n and source[j].isdigit():
                    kind = "NUMBER"
                    i = j
                    while i < n and source[i].isdigit():
                        i += 1
        elif ch.isalpha() or ch == "_":
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
            text = source[start:i]
            kind = text if text in KEYWORDS else "NAME"
        else:
            for punct in PUNCTUATION:
                if source.startswith(punct, i):
                    kind = punct
                    i += len(punct)
                    break
            else:
                raise DimGradError(Diagnostic(line, col, f"unexpected character '{ch}'"))
        tokens.append(Token(kind, source[start:i], line, col))
        col += i - start
    tokens.append(Token("EOF", "", line, col))
    return tokens
