# RQ4: cost of the generated gradients

- 33 gradients. Our optimiser removes 48% of the three-address instructions (1279 to 667).
- Gradient run time relative to the function (GCC -O2), geometric mean: 1.22x (range 0.98x to 2.07x). A central difference costs 2x and returns only the derivative; the gradient function also returns the value.
- Speed-up from our optimiser when GCC also optimises (-O2), geometric mean: 1.24x.
- Speed-up from our optimiser when GCC does not (-O0), geometric mean: 1.59x.
- Against closed forms (13 gradients): largest relative error 4.0e-16 for generated gradients, 4.3e-11 for central differences.

Times are the best of five batches on one machine and vary from run to run; the instruction counts and accuracy figures are deterministic. Functions that take about a nanosecond are at the resolution of the timing loop, so their ratios are rough.

| benchmark | gradient | TAC before | TAC after | function ns | gradient ns | gradient / function | unoptimised TAC, -O2 ns | -O0 ns | unoptimised TAC, -O0 ns |
|---|---|---|---|---|---|---|---|---|---|
| black_scholes | `call_price_grad_spot` | 79 | 40 | 48.5 | 54.2 | 1.12 | 56.8 | 82.8 | 90.0 |
| black_scholes | `call_price_grad_vol` | 79 | 42 | 48.5 | 55.2 | 1.14 | 57.4 | 81.4 | 91.4 |
| black_scholes | `call_price_grad_rate` | 79 | 41 | 48.5 | 54.7 | 1.13 | 56.3 | 84.3 | 92.7 |
| black_scholes | `call_price_grad_expiry` | 79 | 46 | 48.5 | 54.9 | 1.13 | 55.9 | 85.4 | 91.4 |
| chem_rk4 | `concentration_grad_a0` | 128 | 53 | 487.4 | 512.7 | 1.05 | 722.8 | 2129.4 | 4213.5 |
| chem_rk4 | `concentration_grad_k` | 128 | 61 | 487.4 | 610.0 | 1.25 | 723.7 | 2607.9 | 4225.9 |
| climate_ebm | `temperature_grad_T0` | 46 | 24 | 2460.5 | 2488.1 | 1.01 | 2920.7 | 7467.0 | 10963.1 |
| climate_ebm | `temperature_grad_solar` | 46 | 26 | 2460.5 | 2479.5 | 1.01 | 2906.8 | 7634.5 | 10927.0 |
| climate_ebm | `temperature_grad_albedo` | 46 | 27 | 2460.5 | 2453.2 | 1.00 | 2913.3 | 7635.5 | 10969.6 |
| drag_power | `power_grad_v` | 20 | 8 | 0.6 | 0.8 | 1.34 | 1.5 | 4.6 | 10.1 |
| drag_power | `top_speed_grad_p` | 22 | 11 | 9.1 | 11.4 | 1.25 | 13.6 | 19.6 | 30.9 |
| drag_power | `top_speed_grad_area` | 22 | 12 | 9.1 | 11.6 | 1.27 | 13.8 | 19.7 | 27.3 |
| euler_cooling | `cool_grad_T0` | 36 | 12 | 115.8 | 113.1 | 0.98 | 151.7 | 677.9 | 1494.5 |
| euler_cooling | `cool_grad_r` | 36 | 14 | 115.8 | 132.1 | 1.14 | 151.8 | 683.2 | 1459.4 |
| gradient_descent | `f_grad_x` | 14 | 6 | 0.6 | 0.6 | 1.01 | 0.6 | 3.7 | 6.8 |
| gravity | `potential_grad_r` | 16 | 7 | 0.9 | 1.8 | 1.98 | 1.8 | 4.7 | 9.0 |
| gravity | `orbital_speed_grad_r` | 13 | 8 | 2.8 | 4.7 | 1.72 | 4.6 | 7.2 | 9.1 |
| kinetic_energy | `energy_grad_v` | 12 | 4 | 0.6 | 0.6 | 1.00 | 0.7 | 3.3 | 7.2 |
| kinetic_energy | `energy_grad_mass` | 12 | 5 | 0.6 | 0.6 | 1.00 | 0.9 | 4.4 | 7.4 |
| newton_root | `g_grad_x` | 6 | 3 | 0.6 | 0.6 | 1.01 | 0.6 | 2.9 | 3.6 |
| pendulum | `angle_grad_theta0` | 33 | 22 | 1705.0 | 2037.5 | 1.19 | 2132.1 | 4463.0 | 5258.4 |
| pendulum | `angle_grad_len` | 33 | 27 | 1705.0 | 2031.5 | 1.19 | 2132.2 | 4503.4 | 5145.1 |
| pendulum | `period_grad_len` | 17 | 9 | 2.9 | 3.7 | 1.30 | 4.8 | 7.7 | 10.5 |
| projectile_drag | `height_grad_v0` | 42 | 31 | 619.2 | 669.6 | 1.08 | 982.3 | 3022.8 | 4680.1 |
| projectile_drag | `height_grad_k` | 42 | 32 | 619.2 | 734.3 | 1.19 | 972.5 | 3499.9 | 4691.8 |
| range_newton | `miss_grad_v` | 23 | 11 | 4.7 | 5.0 | 1.06 | 12.5 | 11.6 | 26.0 |
| spring_mass | `position_grad_x0` | 34 | 23 | 509.5 | 609.1 | 1.20 | 887.3 | 2185.9 | 3480.4 |
| spring_mass | `position_grad_k` | 34 | 25 | 509.5 | 676.6 | 1.33 | 876.9 | 2460.9 | 3513.8 |
| spring_mass | `total_energy_grad_x` | 26 | 9 | 0.7 | 0.7 | 1.08 | 1.5 | 5.1 | 10.4 |
| spring_mass | `total_energy_grad_v` | 26 | 9 | 0.7 | 0.7 | 1.11 | 1.4 | 5.1 | 10.3 |
| thrust_conversion | `delta_v_grad_impulse` | 10 | 5 | 0.9 | 1.8 | 1.97 | 2.0 | 3.5 | 5.4 |
| thrust_conversion | `position_error_grad_impulse` | 20 | 7 | 0.9 | 1.9 | 2.06 | 2.0 | 4.6 | 10.4 |
| thrust_conversion | `position_error_grad_mass` | 20 | 7 | 0.9 | 1.9 | 2.07 | 1.9 | 4.6 | 10.3 |

## Accuracy against closed forms

| benchmark | gradient | points | generated gradient: largest relative error | central difference: largest relative error |
|---|---|---|---|---|
| drag_power | `power_grad_v` | 8 | 2.1e-16 | 1.6e-11 |
| drag_power | `top_speed_grad_p` | 8 | 4.0e-16 | 3.0e-11 |
| gradient_descent | `f_grad_x` | 3 | 0.0e+00 | 7.3e-12 |
| gravity | `potential_grad_r` | 4 | 0.0e+00 | 4.3e-11 |
| gravity | `orbital_speed_grad_r` | 4 | 3.3e-16 | 2.6e-11 |
| kinetic_energy | `energy_grad_v` | 9 | 0.0e+00 | 3.5e-12 |
| kinetic_energy | `energy_grad_mass` | 9 | 0.0e+00 | 2.6e-11 |
| newton_root | `g_grad_x` | 4 | 0.0e+00 | 2.8e-11 |
| pendulum | `period_grad_len` | 3 | 0.0e+00 | 9.4e-12 |
| range_newton | `miss_grad_v` | 8 | 1.5e-16 | 3.0e-11 |
| spring_mass | `total_energy_grad_x` | 16 | 0.0e+00 | 2.3e-11 |
| spring_mass | `total_energy_grad_v` | 16 | 0.0e+00 | 2.4e-11 |
| thrust_conversion | `delta_v_grad_impulse` | 4 | 1.3e-16 | 4.1e-12 |
