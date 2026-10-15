# RQ2: detection of injected unit errors

185 unit-changing mutants from 14 programs; 182 rejected, 3 accepted. Original programs wrongly rejected: 0.

| category | mutants | rejected | detection rate | error on the mutated line | error in the mutated function |
|---|---|---|---|---|---|
| 1. add/subtract different units | 32 | 31 | 96.9% | 96.8% | 100.0% |
| 2. wrong exponent | 42 | 42 | 100.0% | 85.7% | 100.0% |
| 3. missing or extra conversion constant | 80 | 78 | 97.5% | 76.9% | 98.7% |
| 4. swapped arguments | 14 | 14 | 100.0% | 100.0% | 100.0% |
| 5. wrong variable or unit in a derivative-based update | 17 | 17 | 100.0% | 17.6% | 100.0% |
| all | 185 | 182 | 98.4% | 78.6% | 99.5% |

## Mutants the checker accepted

| benchmark | category | function | line | mutation |
|---|---|---|---|---|
| gradient_descent | 1 | gd_step | 16 | `x - eta * f_grad_x(x)  ->  x - eta` |
| gradient_descent | 3 | gd_step | 16 | `eta  ->  eta * mutant_factor` |
| gradient_descent | 3 | minimize | 22 | `eta  ->  eta * mutant_factor` |
