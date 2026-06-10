# Pump Intensity Sweep Summary (dim=1)

## Configuration
- Sweep range: `1.000e+08` to `1.000e+11` W/cm^2
- Number of points: `20`
- Spacing: `log`
- Workers: `20`
- Mode: `full`
- Materials: `fit`
- High-index material: `sic`
- High-index n/k/n2: `None` / `None` / `None`
- Geometry file: `SiC_optimizations/sic_L3p2um/geometry.json`
- Cavity modes file: `SiC_optimizations/sic_L3p2um/cavity_modes_resonant.json`
- Decay threshold: `0.0001`

## Final Rotation Data

| Pump intensity (W/cm^2) | Final wrapped (deg) | Min (deg) | Max (deg) |
|---:|---:|---:|---:|
| 1.000000e+08 | 0.000366840 | -0.765477149 | 0.757097798 |
| 1.438450e+08 | 0.000446187 | -0.918313757 | 0.907623618 |
| 2.069138e+08 | 0.000545968 | -1.101708702 | 1.087967609 |
| 2.976351e+08 | 0.000667904 | -1.322269603 | 1.303982109 |
| 4.281332e+08 | 0.000821968 | -1.588212560 | 1.562639522 |
| 6.158482e+08 | 0.001018073 | -1.908143452 | 1.872230751 |
| 8.858668e+08 | 0.001262384 | -2.293208016 | 2.242589309 |
| 1.274275e+09 | 0.001578621 | -2.756908816 | 2.685338108 |
| 1.832981e+09 | 0.001981309 | -3.315608201 | 3.214150692 |
| 2.636651e+09 | 0.002504595 | -3.989125216 | 3.845009647 |
| 3.792690e+09 | 0.003189784 | -4.801425619 | 4.596430364 |
| 5.455595e+09 | 0.004097022 | -5.781382133 | 5.511507839 |
| 7.847600e+09 | 0.005298628 | -6.801621697 | 6.561956441 |
| 1.128838e+10 | 0.006918658 | -8.147137941 | 7.955456833 |
| 1.623777e+10 | 0.009114200 | -9.747642035 | 9.465761167 |
| 2.335721e+10 | 0.012106297 | -11.656693823 | 11.231549585 |
| 3.359818e+10 | 0.016248742 | -14.056456416 | 13.395589367 |
| 4.832930e+10 | 0.022018874 | -17.528613336 | 15.926239760 |
| 6.951928e+10 | 0.030144611 | -20.929272956 | 18.980061905 |
| 1.000000e+11 | 0.041761913 | -24.837240966 | 22.650540746 |

## Key Metrics
- Best `|theta_final|`: `0.0417619` deg at `I=1.000000e+11` W/cm^2
- Mean `|theta_final|`: `0.00810463` deg
- Max `|theta_final|`: `0.0417619` deg
- Mean signed `theta_final`: `0.00810463` deg

## Wavelength Targets
- Pump1: `1.869159` um
- Pump2: `1.954511` um
- Probe: `0.824172` um

## Run Parameters
- Resolution: `100`
- Pulse duration: `100.0` fs
- Probe intensity: `50000000.0` W/cm^2
- Pump cutoff: `4.0`

## Fit Models
- Linear vs log-intensity: `theta = 0.0098964*log10(I) + -0.0859111`, `R^2=0.66614`
- Power-law on |theta|: `|theta| = 1.13476e-09*I^0.678802`, `R^2(log)=0.99431`

## Plots

![rotation summary](./rotation_vs_intensity.png)
![dft traces](./dft_traces_vs_intensity.png)
![time-domain traces](./time_domain_traces_vs_intensity.png)

- CSV points file: `rotation_vs_intensity_points.csv`
- JSON sweep report: `../pump_intensity_sweep_report.json`
