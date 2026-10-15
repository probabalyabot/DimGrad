# DimGrad

A source-to-source compiler for a small language with unit annotations. It

- infers and checks the unit of every expression, with exact fractional exponents;
- differentiates functions in forward mode over three-address code and gives every
  derivative the unit of the function divided by the unit of the variable;
- solves the unit a parameter must have when it is declared with `?` (a learning rate, a step size);
- emits C for each function and each gradient.

It implements the design in the base paper, *A Source-to-Source Compiler that Infers and
Checks the Units of Derivatives and Emits C for Functions and Their Gradients*.

```
unit kg, m, s;
unit J = kg*m^2/s^2;

fn energy(mass: kg, v: m/s) -> J {
    return 0.5 * mass * v * v;
}

grad energy wrt v;
```

```
$ python -m dimgrad check benchmarks/kinetic_energy.dg
fn energy(mass: kg, v: m/s) -> J
grad energy wrt v : kg·m/s
```

## Requirements

Python 3.10 or newer; no third-party packages. GCC is needed only to build the generated C
(and for the tests and evaluation scripts that do so). `pip install pytest` to run the tests.

## Usage

```bash
python -m dimgrad check FILE                 # unit report, or unit errors
python -m dimgrad compile FILE -o out.c      # C for every function and gradient
python -m dimgrad run FILE FUNCTION ARGS...  # evaluate with the built-in interpreter
```

`compile` options: `--no-opt` skips the optimiser, `--emit-tac` prints the three-address
code, `--stats` prints instruction counts before and after optimisation, `--harness` adds
a `main()` that reads `function arg...` lines from standard input.

A unit error looks like this:

```
feet_metres.dg:5:20: error: cannot add ft and m
    in expression: reading + correction
    left side:  reading : ft
    right side: correction : m
```

The language is described in [docs/grammar.md](docs/grammar.md).

## How it works

The package has one folder per stage, in pipeline order:

```
dimgrad/
  frontend/   lexer.py, parser.py, ast_nodes.py, unparse.py
  units/      units.py, solver.py, symtab.py, checker.py
  middle/     tac.py, lower.py, interp.py
  diff/       inline.py, autodiff.py, verify.py
  backend/    optimizer.py, emit_c.py, cc.py
  glue/       compiler.py, cli.py, errors.py, fd.py
```

| Stage | File | What it does |
|---|---|---|
| Lexer | `frontend/lexer.py` | tokens with kind, text, line and column |
| Parser | `frontend/parser.py` | recursive descent, builds `ast_nodes.py` trees |
| Symbol table | `units/symtab.py` | nested scopes; records for constants, functions, gradients |
| Unit checker | `units/checker.py`, `solver.py`, `units.py` | unit of every expression; unit equations solved by Gaussian elimination over the rationals |
| Three-address code | `middle/lower.py`, `tac.py` | instructions that each carry a unit; `if`/`for` stay structured |
| Differentiation | `diff/inline.py`, `autodiff.py` | a tangent instruction beside every instruction |
| Unit verifier | `diff/verify.py` | re-derives every unit in the generated code, tangents included |
| Optimiser | `backend/optimizer.py` | constant folding and propagation, algebraic identities, CSE, dead-code elimination |
| C emitter | `backend/emit_c.py` | C with units as comments, optional test harness; `cc.py` runs GCC |
| Driver | `glue/compiler.py`, `cli.py` | runs the stages in order; the `check`, `compile` and `run` commands |

`middle/interp.py` interprets the three-address code and is the oracle the optimiser and
the emitter are tested against. `glue/errors.py` holds the diagnostics every stage raises,
and `glue/fd.py` the central finite differences used by the tests and the evaluation.

## Tests

```bash
python -m pytest -q
```

Covers each stage, golden files for error messages (`tests/golden/errors`; regenerate with
`UPDATE_GOLDEN=1`), every differentiation rule against its closed form, every benchmark
gradient against central finite differences, generated C against the interpreter, and 300
random programs.

## Evaluation

The four research questions of the paper's Section 7. Each script prints a table and writes
it to `evaluation/results/`.

```bash
python -m evaluation.fd_check   # RQ1: gradients against central finite differences
python -m evaluation.mutate     # RQ2: detection of injected unit errors
python -m evaluation.methods    # RQ3: Newton, gradient-descent and Euler updates
python -m evaluation.bench      # RQ4: cost of the gradient, effect of the optimiser
```

The benchmark programs are in `benchmarks/`, with their input grids in `benchmarks/grids.json`.

## Limitations

Scalar reals only: no arrays, `while` loops or recursion. Forward mode only, first
derivatives only. No unit polymorphism. Derivatives where a branch condition flips are
those of the branch taken. A dimensionally consistent program can still be physically
wrong. Angles are unitless. The optimiser folds `0*x` to `0`, which differs from IEEE 754
when `x` is infinite or NaN.
