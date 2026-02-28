# Weighted Bayes + 9-Worker Sweep: Accuracy/Performance Comparison

## What was run
- New optimization: `optimize_cavity_geometry.py` (`bayes`, weighted objective, 6 workers, fit materials).
- New sweep: `pump_intensity_sweep.py`, quasi-1D (`dim=1` path), 9 workers, `resolution=30`, `decay_threshold=1e-5`, 9 log-spaced intensities in `[1e8, 2e12] W/cm^2`.
- Baseline sweep for comparison: same sweep settings, previous baseline geometry/modes (`optimized_geometry_bayes_w6_q40_rerun.json`).

## Optimization Comparison
| Metric | Previous weighted Bayes | New weighted Bayes | Delta |
|---|---:|---:|---:|
| Objective score | 0.058642 | 0.063081 | +0.004439 |
| |rotation| (deg) | 0.090833 | 0.130954 | +0.040121 |
| Quality factor | 0.645605 | 0.481704 | -0.163901 |
| Tail std (deg) | 0.093472 | 0.278477 | +0.185005 |
| eval_count | 56 | 44 | -12 |
| L_cav (um) | 4.4432 | 2.9872 | -1.4560 |
| probe wavelength (um) | 0.8716 | 0.9315 | +0.0599 |
| pump1 wavelength (um) | 1.3422 | 1.3523 | +0.0101 |
| pump2 wavelength (um) | 1.4262 | 1.4390 | +0.0128 |

- Measured wall time (new optimization run): **1176 s** (~19.6 min).
- Measured wall time (new sweep): **329 s** (~5.5 min).
- Measured wall time (baseline sweep): **497 s** (~8.3 min).

## Sweep Comparison (same sweep settings)
| Metric | Baseline geometry | New weighted geometry | Delta |
|---|---:|---:|---:|
| max |theta_rel| (deg) | 0.721722 | 0.088463 | -0.633259 |
| mean |theta_rel| (deg) | 0.091088 | 0.019210 | -0.071877 |
| std(theta_rel) (deg) | 0.224492 | 0.027316 | -0.197176 |
| linear fit slope a (deg/dec) | 0.098106 | 0.015972 | -0.082133 |
| linear fit R^2 | 0.368010 | 0.658820 | +0.290809 |
| power-law exponent p | 0.886219 | 0.513759 | -0.372460 |
| power-law R^2(log) | 0.732581 | 0.999030 | +0.266450 |

### Pointwise final theta_rel
| Pump intensity (W/cm^2) | Baseline theta_rel (deg) | New theta_rel (deg) | New-Baseline (deg) |
|---:|---:|---:|---:|
| 1.000e+08 | -0.000087 | 0.000507 | +0.000594 |
| 3.448e+08 | -0.000118 | 0.000924 | +0.001041 |
| 1.189e+09 | -0.000215 | 0.001728 | +0.001943 |
| 4.101e+09 | -0.000251 | 0.003266 | +0.003517 |
| 1.414e+10 | -0.000021 | 0.005995 | +0.006016 |
| 4.877e+10 | 0.001762 | 0.011147 | +0.009386 |
| 1.682e+11 | 0.012206 | 0.020835 | +0.008629 |
| 5.800e+11 | 0.083408 | 0.040028 | -0.043380 |
| 2.000e+12 | 0.721722 | 0.088463 | -0.633259 |

## Interpretation
- The **new weighted-optimization geometry** improves the optimization objective score and target |rotation| at the optimization point.
- In the sweep, the new geometry gives a **much smoother intensity trend** (`R^2_log` near 1.0) and shorter runtime, but lower high-intensity peak rotation than the baseline geometry.
- The baseline geometry produces larger high-intensity rotation but also stronger long-tail dynamics (runtime penalty and lower fit consistency), suggesting less stable operating behavior under this sweep setup.

## Plots
### New optimization debug plots
![](./bayes_w6_signal_new_epsilon_profile.png)
![](./bayes_w6_signal_new_reflectance_marked.png)
![](./bayes_w6_signal_new_mode_profiles.png)
![](./bayes_w6_signal_new_mode_overlap_matrix.png)

### Sweep plots: new geometry
![](./pump_intensity_sweep_bayes_w6_signal_new_dim1_i9_w9/dim1/rotation_vs_intensity.png)
![](./pump_intensity_sweep_bayes_w6_signal_new_dim1_i9_w9/dim1/dft_traces_vs_intensity.png)
![](./pump_intensity_sweep_bayes_w6_signal_new_dim1_i9_w9/dim1/time_domain_traces_vs_intensity.png)

### Sweep plots: baseline geometry
![](./pump_intensity_sweep_baseline_q40_dim1_i9_w9/dim1/rotation_vs_intensity.png)
![](./pump_intensity_sweep_baseline_q40_dim1_i9_w9/dim1/dft_traces_vs_intensity.png)
![](./pump_intensity_sweep_baseline_q40_dim1_i9_w9/dim1/time_domain_traces_vs_intensity.png)
