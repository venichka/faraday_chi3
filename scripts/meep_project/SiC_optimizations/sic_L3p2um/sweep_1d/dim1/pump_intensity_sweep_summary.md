# Pump Intensity Sweep Summary (dim=1)

## Configuration
- Sweep range: `1.000e+08` to `2.000e+12` W/cm^2
- Number of points: `12`
- Spacing: `log`
- Workers: `12`
- Mode: `full`
- Materials: `fit`
- High-index material: `sic`
- High-index n/k/n2: `None` / `None` / `None`
- Geometry file: `SiC_optimizations/sic_L3p2um/geometry.json`
- Cavity modes file: `SiC_optimizations/sic_L3p2um/cavity_modes.json`
- Decay threshold: `0.0001`

## Final Rotation Data

| Pump intensity (W/cm^2) | Final wrapped (deg) | Min (deg) | Max (deg) |
|---:|---:|---:|---:|
| 1.000000e+08 | 0.000462812 | -3.016647031 | 2.898085946 |
| 2.460383e+08 | 0.000807433 | -4.795984661 | 4.497640998 |
| 6.053485e+08 | 0.001432474 | -7.696889584 | 6.932219255 |
| 1.489389e+09 | 0.002756203 | -12.479399765 | 10.617546553 |
| 3.664468e+09 | 0.005318543 | -20.442555553 | 16.114411093 |
| 9.015994e+09 | 0.011871432 | -33.440117910 | 24.221020291 |
| 2.218280e+10 | 0.026194433 | -58.603790367 | 51.345775541 |
| 5.457818e+10 | 0.070244888 | -89.543664917 | 89.872493151 |
| 1.342832e+11 | 0.216455148 | -89.822864281 | 89.787844546 |
| 3.303882e+11 | 0.906707369 | -89.973909347 | 89.968430608 |
| 8.128816e+11 | 5.825805650 | -89.615081733 | 89.975434794 |
| 2.000000e+12 | 16.551001065 | -89.884448236 | 89.962858521 |

## Key Metrics
- Best `|theta_final|`: `16.551` deg at `I=2.000000e+12` W/cm^2
- Mean `|theta_final|`: `1.96825` deg
- Max `|theta_final|`: `16.551` deg
- Mean signed `theta_final`: `1.96825` deg

## Wavelength Targets
- Pump1: `1.656531` um
- Pump2: `1.723221` um
- Probe: `0.824172` um

## Run Parameters
- Resolution: `100`
- Pulse duration: `100.0` fs
- Probe intensity: `50000000.0` W/cm^2
- Pump cutoff: `4.0`

## Fit Models
- Linear vs log-intensity: `theta = 2.16491*log10(I) + -20.0067`, `R^2=0.39070`
- Power-law on |theta|: `|theta| = 7.37508e-13*I^1.05165`, `R^2(log)=0.96471`

## Plots

![rotation summary](./rotation_vs_intensity.png)
![dft traces](./dft_traces_vs_intensity.png)
![time-domain traces](./time_domain_traces_vs_intensity.png)

- CSV points file: `rotation_vs_intensity_points.csv`
- JSON sweep report: `../pump_intensity_sweep_report.json`
