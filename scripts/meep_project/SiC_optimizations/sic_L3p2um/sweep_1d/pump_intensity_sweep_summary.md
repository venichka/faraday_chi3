# Pump Intensity Sweep - Global Summary

## Configuration
- Dimensions: `[1]`
- Workers requested/effective: `12` / `12`
- Range scale: `log`
- Intensity range: `1.000e+08` to `2.000e+12` W/cm^2
- Number of points: `12`
- Mode/materials: `full` / `fit`
- High-index material n/k/n2: `sic` / `None` / `None` / `None`

## Dimension Comparison

| dim | best |theta| (deg) | intensity at best (W/cm^2) | linear slope a (deg/dec) | power exponent p |
|---:|---:|---:|---:|---:|
| 1 | 16.551 | 2.000000e+12 | 2.16491 | 1.05165 |

## Per-Dimension Artifacts
- dim=1
  - Markdown summary: `dim1/pump_intensity_sweep_summary.md`
  - CSV points: `dim1/rotation_vs_intensity_points.csv`
  - Rotation plot: `dim1/rotation_vs_intensity.png`
  - DFT traces: `dim1/dft_traces_vs_intensity.png`
  - TD traces: `dim1/time_domain_traces_vs_intensity.png`

## Embedded Plots

### dim=1
![dim1 rotation](./dim1/rotation_vs_intensity.png)
![dim1 dft traces](./dim1/dft_traces_vs_intensity.png)
![dim1 td traces](./dim1/time_domain_traces_vs_intensity.png)
