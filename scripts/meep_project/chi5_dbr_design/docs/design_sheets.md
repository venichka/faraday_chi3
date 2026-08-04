# Fabrication spec sheets — chi5 DBR design campaign finalists

Generated from `runs/s2_fdtd/s2_result.json` (1D FDTD, carrier-averaged,
pulse-integrated objective at I_pump = 1e12 W/cm^2, 100 fs intensity FWHM).

All stacks are SiN/SiO2 on an SiO2 substrate, deposited from the incident (air) face
inward. Constraints honoured: <= 6 mirror pairs per side, every layer >= 80 nm,
total stack <= 12 um.

**Read the contrast column.** `theta` is how much rotation the design makes;
`contrast = theta / fringe` is whether it can be separated from the coherent carrier
fringe without an interferometrically stable delay line. They do not rank together.

## Summary

| design | theta_chi5 | vs fabricated | contrast | DoLP | probe (nm) | pump1/pump2 (nm) | Delta (1/um) | stack (um) |
|---|---|---|---|---|---|---|---|---|
| cand07 | 0.08156° | 11.44× | **0.10** | 0.990 | 801.6 | 1395.4 / 1423.2 | 0.0140 | 11.89 |
| cand13 | 0.07631° | 10.70× | **0.57** | 0.994 | 792.7 | 1471.4 / 1522.9 | 0.0230 | 10.98 |
| cand16 | 0.04553° | 6.39× | **1.63** | 0.994 | 790.0 | 1511.7 / 1544.4 | 0.0140 | 10.58 |
| cand15 | 0.05855° | 8.21× | **1.32** | 0.995 | 790.5 | 1674.1 / 1741.2 | 0.0230 | 9.85 |
| baseline | 0.00713° | 1.00× | **0.09** | 0.996 | 800.1 | 1492.6 / 1525.1 | 0.0143 | 9.38 |

---

## cand07

* mirror pairs: **5 left / 2 right**  (asymmetric)
* SiN layer **464.5 nm**, SiO2 layer **122.7 nm**  (t_lo/t_hi = **0.26**)
* cavity (SiN): **7781.7 nm**
* total deposited: **11.892 um**

**Operating point** — probe **801.6 nm**, pumps **1395.4 / 1423.2 nm** (separation 27.8 nm, Delta = 0.0140 /um), balanced sigma+ sigma-, counter-rotating.

**Simulated (1D)** — theta_chi5 = **0.08156°** (11.44× the fabricated design), carrier-fringe amplitude 0.83296°, contrast **0.10**, DoLP 0.990.

**Stage-3 validation**

* **intensity scaling** local log-log slopes 1.29, 1.56, 1.67, 1.47 (chi5 => 2); 0.717° at 4.0e+12 W/cm² (DoLP 0.858)
* **tolerance sigma = 3.0 nm**: median 98%, 10th pct 46%, worst 21% of nominal (n=12)
* **tolerance sigma = 5.0 nm**: median 161%, 10th pct 37%, worst 2% of nominal (n=12)
* **3D**: theta_chi5 = **0.47652°** (5.84× the 1D value), contrast 0.13, DoLP 0.233

| # | material | thickness (nm) | cumulative (nm) |
|---|---|---|---|
| 1 | SiN | 464.5 | 464.5 |
| 2 | SiO2 | 122.7 | 587.2 |
| 3 | SiN | 464.5 | 1051.7 |
| 4 | SiO2 | 122.7 | 1174.4 |
| 5 | SiN | 464.5 | 1638.9 |
| 6 | SiO2 | 122.7 | 1761.5 |
| 7 | SiN | 464.5 | 2226.0 |
| 8 | SiO2 | 122.7 | 2348.7 |
| 9 | SiN | 464.5 | 2813.2 |
| 10 | SiO2 | 122.7 | 2935.9 |
| 11 | SiN  (CAVITY) | 7781.7 | 10717.6 |
| 12 | SiO2 | 122.7 | 10840.3 |
| 13 | SiN | 464.5 | 11304.7 |
| 14 | SiO2 | 122.7 | 11427.4 |
| 15 | SiN | 464.5 | 11891.9 |

---

## cand13

* mirror pairs: **5 left / 4 right**  (asymmetric)
* SiN layer **420.8 nm**, SiO2 layer **158.7 nm**  (t_lo/t_hi = **0.38**)
* cavity (SiN): **5762.2 nm**
* total deposited: **10.978 um**

**Operating point** — probe **792.7 nm**, pumps **1471.4 / 1522.9 nm** (separation 51.5 nm, Delta = 0.0230 /um), balanced sigma+ sigma-, counter-rotating.

**Simulated (1D)** — theta_chi5 = **0.07631°** (10.70× the fabricated design), carrier-fringe amplitude 0.13298°, contrast **0.57**, DoLP 0.994.

**Stage-3 validation**

* **intensity scaling** local log-log slopes 2.03, 2.03, 2.04, 2.04 (chi5 => 2); 1.288° at 4.0e+12 W/cm² (DoLP 0.904)
* **tolerance sigma = 3.0 nm**: median 88%, 10th pct 74%, worst 72% of nominal (n=12)
* **tolerance sigma = 5.0 nm**: median 115%, 10th pct 44%, worst 33% of nominal (n=12)
* **3D**: theta_chi5 = **0.56249°** (7.37× the 1D value), contrast 0.94, DoLP 0.757

| # | material | thickness (nm) | cumulative (nm) |
|---|---|---|---|
| 1 | SiN | 420.8 | 420.8 |
| 2 | SiO2 | 158.7 | 579.6 |
| 3 | SiN | 420.8 | 1000.4 |
| 4 | SiO2 | 158.7 | 1159.1 |
| 5 | SiN | 420.8 | 1580.0 |
| 6 | SiO2 | 158.7 | 1738.7 |
| 7 | SiN | 420.8 | 2159.5 |
| 8 | SiO2 | 158.7 | 2318.3 |
| 9 | SiN | 420.8 | 2739.1 |
| 10 | SiO2 | 158.7 | 2897.8 |
| 11 | SiN  (CAVITY) | 5762.2 | 8660.0 |
| 12 | SiO2 | 158.7 | 8818.7 |
| 13 | SiN | 420.8 | 9239.6 |
| 14 | SiO2 | 158.7 | 9398.3 |
| 15 | SiN | 420.8 | 9819.1 |
| 16 | SiO2 | 158.7 | 9977.9 |
| 17 | SiN | 420.8 | 10398.7 |
| 18 | SiO2 | 158.7 | 10557.4 |
| 19 | SiN | 420.8 | 10978.3 |

---

## cand16

* mirror pairs: **6 left / 2 right**  (asymmetric)
* SiN layer **360.8 nm**, SiO2 layer **320.5 nm**  (t_lo/t_hi = **0.89**)
* cavity (SiN): **5131.3 nm**
* total deposited: **10.581 um**

**Operating point** — probe **790.0 nm**, pumps **1511.7 / 1544.4 nm** (separation 32.7 nm, Delta = 0.0140 /um), balanced sigma+ sigma-, counter-rotating.

**Simulated (1D)** — theta_chi5 = **0.04553°** (6.39× the fabricated design), carrier-fringe amplitude 0.02794°, contrast **1.63**, DoLP 0.994.

**Stage-3 validation**

* **intensity scaling** local log-log slopes 1.64, 1.82, 1.86, 1.70 (chi5 => 2); 0.539° at 4.0e+12 W/cm² (DoLP 0.901)
* **tolerance sigma = 3.0 nm**: median 81%, 10th pct 61%, worst 39% of nominal (n=12)
* **tolerance sigma = 5.0 nm**: median 102%, 10th pct 63%, worst 17% of nominal (n=12)
* **3D**: theta_chi5 = **0.20625°** (4.53× the 1D value), contrast 1.51, DoLP 0.699

| # | material | thickness (nm) | cumulative (nm) |
|---|---|---|---|
| 1 | SiN | 360.8 | 360.8 |
| 2 | SiO2 | 320.5 | 681.3 |
| 3 | SiN | 360.8 | 1042.0 |
| 4 | SiO2 | 320.5 | 1362.5 |
| 5 | SiN | 360.8 | 1723.3 |
| 6 | SiO2 | 320.5 | 2043.8 |
| 7 | SiN | 360.8 | 2404.5 |
| 8 | SiO2 | 320.5 | 2725.0 |
| 9 | SiN | 360.8 | 3085.8 |
| 10 | SiO2 | 320.5 | 3406.3 |
| 11 | SiN | 360.8 | 3767.0 |
| 12 | SiO2 | 320.5 | 4087.5 |
| 13 | SiN  (CAVITY) | 5131.3 | 9218.9 |
| 14 | SiO2 | 320.5 | 9539.4 |
| 15 | SiN | 360.8 | 9900.1 |
| 16 | SiO2 | 320.5 | 10220.6 |
| 17 | SiN | 360.8 | 10581.4 |

---

## cand15

* mirror pairs: **5 left / 6 right**  (asymmetric)
* SiN layer **395.5 nm**, SiO2 layer **207.0 nm**  (t_lo/t_hi = **0.52**)
* cavity (SiN): **3224.2 nm**
* total deposited: **9.851 um**

**Operating point** — probe **790.5 nm**, pumps **1674.1 / 1741.2 nm** (separation 67.0 nm, Delta = 0.0230 /um), balanced sigma+ sigma-, counter-rotating.

**Simulated (1D)** — theta_chi5 = **0.05855°** (8.21× the fabricated design), carrier-fringe amplitude 0.04444°, contrast **1.32**, DoLP 0.995.

**Stage-3 validation**

* **intensity scaling** local log-log slopes 2.00, 2.04, 2.07, 2.09 (chi5 => 2); 1.049° at 4.0e+12 W/cm² (DoLP 0.925)
* **tolerance sigma = 3.0 nm**: median 94%, 10th pct 72%, worst 70% of nominal (n=12)
* **tolerance sigma = 5.0 nm**: median 88%, 10th pct 66%, worst 60% of nominal (n=12)
* **3D**: theta_chi5 = **0.37008°** (6.32× the 1D value), contrast 0.59, DoLP 0.862

| # | material | thickness (nm) | cumulative (nm) |
|---|---|---|---|
| 1 | SiN | 395.5 | 395.5 |
| 2 | SiO2 | 207.0 | 602.5 |
| 3 | SiN | 395.5 | 997.9 |
| 4 | SiO2 | 207.0 | 1204.9 |
| 5 | SiN | 395.5 | 1600.4 |
| 6 | SiO2 | 207.0 | 1807.4 |
| 7 | SiN | 395.5 | 2202.9 |
| 8 | SiO2 | 207.0 | 2409.9 |
| 9 | SiN | 395.5 | 2805.4 |
| 10 | SiO2 | 207.0 | 3012.3 |
| 11 | SiN  (CAVITY) | 3224.2 | 6236.5 |
| 12 | SiO2 | 207.0 | 6443.5 |
| 13 | SiN | 395.5 | 6839.0 |
| 14 | SiO2 | 207.0 | 7046.0 |
| 15 | SiN | 395.5 | 7441.5 |
| 16 | SiO2 | 207.0 | 7648.5 |
| 17 | SiN | 395.5 | 8043.9 |
| 18 | SiO2 | 207.0 | 8250.9 |
| 19 | SiN | 395.5 | 8646.4 |
| 20 | SiO2 | 207.0 | 8853.4 |
| 21 | SiN | 395.5 | 9248.9 |
| 22 | SiO2 | 207.0 | 9455.9 |
| 23 | SiN | 395.5 | 9851.3 |

---

## baseline  (the fabricated reference)

* mirror pairs: **3 left / 3 right**
* SiN layer **237.5 nm**, SiO2 layer **344.2 nm**  (t_lo/t_hi = **1.45**)
* cavity (SiN): **5893.7 nm**
* total deposited: **9.384 um**

**Operating point** — probe **800.1 nm**, pumps **1492.6 / 1525.1 nm** (separation 32.5 nm, Delta = 0.0143 /um), balanced sigma+ sigma-, counter-rotating.

**Simulated (1D)** — theta_chi5 = **0.00713°** (1.00× the fabricated design), carrier-fringe amplitude 0.08012°, contrast **0.09**, DoLP 0.996.

**Stage-3 validation**

* **intensity scaling** local log-log slopes 2.28, 2.04, 1.92, 1.71 (chi5 => 2); 0.088° at 4.0e+12 W/cm² (DoLP 0.935)
* **tolerance sigma = 3.0 nm**: median 85%, 10th pct 45%, worst 15% of nominal (n=12)
* **tolerance sigma = 5.0 nm**: median 137%, 10th pct 28%, worst 12% of nominal (n=12)
* **3D**: theta_chi5 = **0.06631°** (9.30× the 1D value), contrast 0.15, DoLP 0.654

| # | material | thickness (nm) | cumulative (nm) |
|---|---|---|---|
| 1 | SiN | 237.5 | 237.5 |
| 2 | SiO2 | 344.2 | 581.7 |
| 3 | SiN | 237.5 | 819.2 |
| 4 | SiO2 | 344.2 | 1163.4 |
| 5 | SiN | 237.5 | 1400.9 |
| 6 | SiO2 | 344.2 | 1745.1 |
| 7 | SiN  (CAVITY) | 5893.7 | 7638.8 |
| 8 | SiO2 | 344.2 | 7983.1 |
| 9 | SiN | 237.5 | 8220.5 |
| 10 | SiO2 | 344.2 | 8564.8 |
| 11 | SiN | 237.5 | 8802.2 |
| 12 | SiO2 | 344.2 | 9146.5 |
| 13 | SiN | 237.5 | 9384.0 |
