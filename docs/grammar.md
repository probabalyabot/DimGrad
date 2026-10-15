# The DimGrad language

A small imperative language with unit annotations: scalar reals, functions, local
variables, `if`/`else` and `for` loops with fixed integer bounds. No `while`, recursion,
pointers or arrays.

## Grammar

```
program    := (unit_decl | const_decl | function | grad_decl)*

unit_decl  := "unit" NAME ("," NAME)* ";"            base units
            | "unit" NAME "=" unit_expr ";"          a named derived unit
const_decl := "const" NAME ":" unit_expr "=" "-"? NUMBER ";"
function   := "fn" NAME "(" params? ")" "->" unit_expr block
params     := param ("," param)*
param      := NAME ":" (unit_expr | "?")
grad_decl  := "grad" NAME "wrt" NAME ";"

block      := "{" statement* "}"
statement  := "let" NAME "=" expr ";"
            | NAME "=" expr ";"
            | "if" "(" cond ")" block ("else" block)?
            | "for" NAME "in" INT ".." INT block
            | "return" expr ";"
cond       := expr ("<" | ">" | "<=" | ">=" | "==" | "!=") expr

expr       := term (("+" | "-") term)*
term       := unary (("*" | "/") unary)*
unary      := "-" unary | power
power      := atom ("^" "-"? INT)?
atom       := NUMBER | NAME | NAME "(" (expr ("," expr)*)? ")" | "(" expr ")"

unit_expr  := unit_term (("*" | "/") unit_term)*
unit_term  := "1" | NAME ("^" unit_exp)?
unit_exp   := "-"? INT | "(" "-"? INT "/" INT ")"
```

Comments run from `//` to the end of the line. `INT` loop bounds may be negative.

### Changes from the provisional grammar in the paper

| Addition | Why |
|---|---|
| `?` as a parameter's unit | Section 5.4 uses it, the grammar did not have it |
| Fractional unit exponents, `s^(1/2)` | Black-Scholes volatility is `1/yr^(1/2)` |
| Unary minus; `<=`, `>=`, `!=`; negative exponents | Needed by ordinary formulas |
| Calls with no arguments | Harmless generalisation |

## Semantics

**Units.** A unit is a product of base units raised to exact rational powers. Unit names
live in their own namespace, so a variable may be called `m` while `m` is also a unit.
A unit must be declared before it is used in another declaration.

**Unit rules** (Section 4.3 of the paper):

| Construct | Rule |
|---|---|
| literal | `0` takes the unit its context requires; every other literal is unitless |
| `a + b`, `a - b` | units must be equal |
| `a * b`, `a / b` | exponents add / subtract |
| `a ^ n` | exponents scale by the integer `n` |
| `sqrt(a)`, `cbrt(a)` | exponents scale by 1/2, 1/3 |
| `sin cos exp log normcdf` | argument and result are unitless |
| comparison | both sides must have equal units |
| `let` | the variable takes the unit of its value |
| assignment | the value must have the variable's unit |
| `return`, call | must match the declared return / parameter units |

`let` is block-scoped and an assignment can never change a variable's unit, so a variable
has the same unit after both branches of an `if` and on every iteration of a loop.

**Placeholders.** A parameter declared `name: ?` has its unit solved from how it is used.
It is an error if nothing determines it. There is no unit polymorphism: the solved unit is
fixed, and callers must match it.

**Names.** A name may not be redeclared while it is visible (no shadowing). Parameters,
constants and loop variables cannot be assigned. Functions may be called before their
definition. Recursion is an error.

**Loops.** `for i in lo..hi` runs `i = lo, ..., hi-1`. The loop variable is a unitless value.

**Return.** `return` must be the last statement of the function, or the last statement of
both branches of a final `if`/`else`. Every path must return.

**Gradients.** `grad f wrt x;` makes the compiler generate the derivative of `f` with
respect to parameter `x`. It also declares a function `f_grad_x` with the same parameters
as `f` and the unit `[f]/[x]`, which other functions may call, for example in a Newton step:

```
fn newton_step(x: m, a: m^2) -> m {
    return x - g(x, a) / g_grad_x(x, a);
}
```

A function that uses a gradient function cannot itself be differentiated (no second
derivatives). At a point where a branch condition flips, the derivative is that of the
branch taken.

## Generated C

For `fn f(a: ..., b: ...)` and `grad f wrt a;` the compiler emits

```c
double f(double a, double b);
double f_grad_a(double a, double b, double *value);   /* returns df/da; stores f in *value unless NULL */
```

Named constants become `static const double`. A name that is reserved in C or `<math.h>`
gets a trailing underscore.
