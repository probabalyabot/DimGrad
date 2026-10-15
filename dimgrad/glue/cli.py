"""Command line: python -m dimgrad {check,compile,run} FILE ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..backend.emit_c import emit_c
from ..middle import tac
from ..middle.interp import evaluate
from .compiler import check_source, compile_source
from .errors import DimGradError


def _stats_table(compiled) -> str:
    rows = [("function", "before", "after")]
    rows += [(name, str(before), str(after)) for name, (before, after) in compiled.stats.items()]
    width = max(len(row[0]) for row in rows)
    return "\n".join(f"{name:<{width}}  {before:>6}  {after:>6}" for name, before, after in rows)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="dimgrad", description="Unit-checking differentiating compiler.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="check units and report function and gradient units")
    p_check.add_argument("file")

    p_compile = sub.add_parser("compile", help="emit C for every function and gradient")
    p_compile.add_argument("file")
    p_compile.add_argument("-o", "--output", help="write the C here instead of standard output")
    p_compile.add_argument("--no-opt", action="store_true", help="skip the optimiser")
    p_compile.add_argument("--harness", action="store_true", help="add a main() that evaluates functions from stdin")
    p_compile.add_argument("--emit-tac", action="store_true", help="print the three-address code")
    p_compile.add_argument("--stats", action="store_true", help="print instruction counts before and after optimisation")

    p_run = sub.add_parser("run", help="evaluate a function or gradient with the built-in interpreter")
    p_run.add_argument("file")
    p_run.add_argument("function")
    p_run.add_argument("args", nargs="*", type=float)
    p_run.add_argument("--no-opt", action="store_true")

    args = parser.parse_args(argv)
    path = Path(args.file)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"dimgrad: cannot read {args.file}: {exc.strerror}", file=sys.stderr)
        return 2

    try:
        if args.command == "check":
            print(check_source(source).report())
            return 0

        compiled = compile_source(source, optimize=not args.no_opt)
        if args.command == "run":
            if args.function not in compiled.program.functions:
                print(f"dimgrad: no function '{args.function}' in {args.file}", file=sys.stderr)
                return 2
            func = compiled.program.functions[args.function]
            if len(args.args) != len(func.params):
                print(f"dimgrad: '{func.name}' expects {len(func.params)} argument(s): "
                      f"{', '.join(func.params)}", file=sys.stderr)
                return 2
            result = evaluate(compiled.program, func.name, args.args)
            fmt = compiled.program.fmt
            if func.kind == "grad":
                print(f"d {func.of} / d {func.wrt} = {result[0]!r} {fmt(func.ret_unit)}")
                print(f"{func.of} = {result[1]!r} {fmt(func.value_unit)}")
            else:
                print(f"{func.name} = {result!r} {fmt(func.ret_unit)}")
            return 0

        c_text = emit_c(compiled.program, path.name, harness=args.harness)
        # With -o the C goes to the file and the listings to stdout; otherwise
        # the C owns stdout and the listings go to stderr.
        info = sys.stdout if args.output else sys.stderr
        if args.emit_tac:
            print(tac.format_program(compiled.program), file=info)
        if args.stats:
            print(_stats_table(compiled), file=info)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(c_text, encoding="utf-8")
        else:
            sys.stdout.write(c_text)
        return 0
    except DimGradError as exc:
        print(exc.format(path.name), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
