# TiO2 vs Si3N4 for Faraday-like Pump-Induced Probe Rotation in Meep

## Scope
- Pump wavelengths: `1.3–1.7 um`
- Probe wavelength: `0.8 um` (or `0.85–0.95 um`)
- Structure: DBR cavity with isotropic Kerr (`chi3`) medium
- Goal here: assess replacing Si3N4/SiN with TiO2 for the nonlinear cavity medium

## 1. Is TiO2 available in Meep materials library in the required range?

Short answer: **not as a built-in dielectric material** in the current Meep Python materials module.

Evidence:
- Local environment check (`meep.materials`) shows `TiO2` is absent, while `SiN`, `Si3N4_NIR`, `SiO2`, and metallic `Ti` entries exist.
- Meep materials docs/source list common dielectrics and metals; Ti entries are metallic, not dielectric TiO2.

Practical implication:
- For TiO2 you should use a **custom dispersive fit** from `n,k` data (your existing `material_fit.py` path already supports this).
- Constant-index fallback can be used for quick scans, but dispersive fitting is preferred for broadband pump/probe optimization.

## 2. If TiO2 is not built-in, what refractive-index parameters are realistic?

Two realistic options (different physics assumptions):

1. **Amorphous/polycrystalline integrated-photonics TiO2 film (recommended for fabrication realism)**
- Crystalline TiO2 waveguide platform reports `n = 2.31` at `1550 nm` from ellipsometry, with very low loss over about `570–1600 nm`.
- This is close to your pump/probe region but does not fully cover `1.7 um` in that dataset.

2. **Bulk rutile TiO2 dispersion (covers wider NIR but anisotropic crystal)**
- RefractiveIndex.INFO TiO2 pages include rutile datasets/formulas and also thin-film datasets up to about `1.69 um` (Sarkar 2019 listing).
- Use with caution: bulk rutile is anisotropic (`n_o != n_e`), while your model currently assumes isotropic medium for Kerr response.

Recommended modeling path in this project:
- Use TiO2 `n,k` CSV over at least `0.75–1.75 um` and fit with your existing Lorentz pipeline.
- If only constant index is needed for a quick test, start with `n ~ 2.3` and treat as a temporary placeholder.

## 3. What n2 should be used for TiO2 in the nonlinear model?

Literature shows a wide spread for integrated TiO2 at telecom:
- **Optics Express 2018 (integrated crystalline TiO2 waveguides):** `n2 = (2.3–3.6) x 10^-18 m^2/W` at `~1550 nm`.
- **Optics Express 2013 (TiO2 waveguides):** `n2 ~ 0.16 x 10^-18 m^2/W` at `1565 nm`.

So TiO2 `n2` is highly process/phase dependent (more than an order of magnitude spread).

Working recommendation for optimization:
- Use a **robust range**, not one fixed number:
  - conservative: `0.2 x 10^-18 m^2/W`
  - nominal: `1.0 x 10^-18 m^2/W`
  - optimistic: `3.0 x 10^-18 m^2/W`
- Re-rank geometries by performance stability across this `n2` range.

Notes vs current SiN setup:
- Your current code uses `n2_sin = 5.0 x 10^-19 m^2/W` (hardcoded scale-up from `2.5 x 10^-19`).
- 2018 TiO2 literature values can exceed typical Si3N4 by roughly ~1 order, but not universally (depends on film/crystal quality and mode overlap extraction).

## 4. What should be the effective chi3 tensor form in Meep?

Current code:
- `mat_sin.E_chi3_diag = mp.Vector3(x, x, x)`

Interpretation in Meep:
- Meep’s built-in Kerr nonlinearity is an **instantaneous isotropic model** (`P ~ chi3 |E|^2 E` form).
- `E_chi3_diag` supports diagonal component scaling, but does **not** represent a full general 4th-rank anisotropic `chi^(3)_{ijkl}` with arbitrary cross-couplings.

Therefore:
1. **If TiO2 is treated as isotropic amorphous/polycrystalline film**:
- Using equal diagonal elements `(x, x, x)` is consistent with the model assumptions.

2. **If TiO2 is crystalline (rutile/anatase) and orientation-sensitive effects matter**:
- Real `chi3` tensor elements differ and can have different signs/magnitudes (anisotropic nonlinear response).
- The simple equal-diagonal Meep model is then an approximation and may miss polarization-coupling details.

## 5. TiO2 vs Si3N4 quick comparison

| Item | Si3N4 (current project baseline) | TiO2 (candidate) |
|---|---|---|
| Meep built-in dielectric | Yes (`SiN`, `Si3N4_NIR`) | No dedicated dielectric TiO2 entry |
| Typical linear index (NIR) | ~2.0 (model dependent) | ~2.3 at 1550 nm (reported integrated TiO2 platform) |
| Nonlinear index `n2` (telecom, representative) | ~`0.2–0.3 x 10^-18 m^2/W` often cited | reported from `0.16` up to `2.3–3.6 x 10^-18 m^2/W` |
| Loss window relevance to this project | generally good in your pump/probe bands | reported low-loss windows can cover probe + much of pump band; verify to 1.7 um for chosen film dataset |
| Tensor/isotropy concern | usually modeled isotropic in integrated SiN | stronger anisotropy risk if crystalline TiO2 is used |

## 6. Concrete next implementation steps (compatible with current pipeline)

1. Add TiO2 `n,k` CSV and fit via existing `material_fit.py`.
2. Extend material selection to allow `high_index_material = TiO2` while keeping DBR low-index layer as SiO2.
3. Replace hardcoded `n2_sin` with configurable `n2_high_index` and run sensitivity over `[0.2e-18, 1.0e-18, 3.0e-18]`.
4. Keep `E_chi3_diag = (x,x,x)` for isotropic-film assumption; document this assumption explicitly in optimization reports.

## Sources

- Meep materials documentation: https://meep.readthedocs.io/en/latest/Materials/
- Meep nonlinear units and `chi3` conventions: https://meep.readthedocs.io/en/latest/Units_and_Nonlinearity/
- Meep materials source (`materials.py`): https://raw.githubusercontent.com/NanoComp/meep/master/python/materials.py
- Guan et al., Optics Express 2018 (integrated crystalline TiO2; `n`, loss window, and `n2` range): https://pmc.ncbi.nlm.nih.gov/articles/PMC6058206/
- Zhang et al., Optics Express 2013 (TiO2 waveguides; extracted `n2` at 1565 nm): https://pubmed.ncbi.nlm.nih.gov/23938657/
- TiO2 refractive-index datasets and ranges (including rutile/thin-film listings): https://refractiveindex.info/?book=TiO2&page=n2&shelf=main
