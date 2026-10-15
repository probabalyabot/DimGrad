"""Diagnostics shared by every compiler stage."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Diagnostic:
    line: int
    col: int
    message: str
    notes: list[str] = field(default_factory=list)

    def format(self, filename: str = "<input>") -> str:
        text = f"{filename}:{self.line}:{self.col}: error: {self.message}"
        for note in self.notes:
            text += f"\n    {note}"
        return text


class DimGradError(Exception):
    """One or more errors in the user's program."""

    def __init__(self, diagnostics):
        if isinstance(diagnostics, Diagnostic):
            diagnostics = [diagnostics]
        self.diagnostics = sorted(diagnostics, key=lambda d: (d.line, d.col))
        super().__init__(self.format())

    def format(self, filename: str = "<input>") -> str:
        return "\n".join(d.format(filename) for d in self.diagnostics)
