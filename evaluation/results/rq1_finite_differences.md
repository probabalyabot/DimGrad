# RQ1: gradients against central finite differences

33 gradients from 14 programs, 432 points, evaluated with GCC-compiled C. Tolerance 1e-06. Largest relative error 2.55e-08; 0 gradient(s) outside tolerance.

| benchmark | gradient | unit of the derivative | points | largest relative error | within tolerance |
|---|---|---|---|---|---|
| black_scholes | `call_price_grad_spot` | 1 | 24 | 9.84e-09 | yes |
| black_scholes | `call_price_grad_vol` | usd·yr^(1/2) | 24 | 1.46e-08 | yes |
| black_scholes | `call_price_grad_rate` | usd·yr | 24 | 2.20e-08 | yes |
| black_scholes | `call_price_grad_expiry` | usd/yr | 24 | 4.00e-10 | yes |
| chem_rk4 | `concentration_grad_a0` | 1 | 12 | 1.59e-10 | yes |
| chem_rk4 | `concentration_grad_k` | mol^2·s/m^6 | 12 | 3.03e-09 | yes |
| climate_ebm | `temperature_grad_T0` | 1 | 24 | 3.61e-10 | yes |
| climate_ebm | `temperature_grad_solar` | s^3·K/kg | 24 | 2.82e-09 | yes |
| climate_ebm | `temperature_grad_albedo` | K | 24 | 1.55e-08 | yes |
| drag_power | `power_grad_v` | kg·m/s^2 | 8 | 1.65e-11 | yes |
| drag_power | `top_speed_grad_p` | s^2/(kg·m) | 8 | 3.05e-11 | yes |
| drag_power | `top_speed_grad_area` | 1/(m·s) | 8 | 4.81e-11 | yes |
| euler_cooling | `cool_grad_T0` | 1 | 8 | 8.62e-10 | yes |
| euler_cooling | `cool_grad_r` | K·s | 8 | 3.42e-09 | yes |
| gradient_descent | `f_grad_x` | kg·m/s^2 | 3 | 7.33e-12 | yes |
| gravity | `potential_grad_r` | kg·m/s^2 | 4 | 4.26e-11 | yes |
| gravity | `orbital_speed_grad_r` | 1/s | 4 | 2.63e-11 | yes |
| kinetic_energy | `energy_grad_v` | kg·m/s | 9 | 3.49e-12 | yes |
| kinetic_energy | `energy_grad_mass` | m^2/s^2 | 9 | 2.61e-11 | yes |
| newton_root | `g_grad_x` | m | 4 | 2.75e-11 | yes |
| pendulum | `angle_grad_theta0` | 1 | 18 | 5.56e-09 | yes |
| pendulum | `angle_grad_len` | 1/m | 18 | 7.31e-10 | yes |
| pendulum | `period_grad_len` | s/m | 3 | 9.43e-12 | yes |
| projectile_drag | `height_grad_v0` | s | 18 | 1.47e-10 | yes |
| projectile_drag | `height_grad_k` | m^2 | 18 | 2.55e-08 | yes |
| range_newton | `miss_grad_v` | s | 8 | 3.04e-11 | yes |
| spring_mass | `position_grad_x0` | 1 | 16 | 3.67e-10 | yes |
| spring_mass | `position_grad_k` | m·s^2/kg | 16 | 5.18e-10 | yes |
| spring_mass | `total_energy_grad_x` | kg·m/s^2 | 16 | 2.29e-11 | yes |
| spring_mass | `total_energy_grad_v` | kg·m/s | 16 | 2.44e-11 | yes |
| thrust_conversion | `delta_v_grad_impulse` | m/(s^2·lbf) | 4 | 4.13e-12 | yes |
| thrust_conversion | `position_error_grad_impulse` | m/(s·lbf) | 8 | 1.56e-11 | yes |
| thrust_conversion | `position_error_grad_mass` | m/kg | 8 | 4.33e-11 | yes |
