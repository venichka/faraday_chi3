"""Regression tests for the chi5_dbr_design campaign helpers.

Meep is NOT required (chi5_dbr_design.common imports tmm lazily), matching the repo rule that
the math/parsing helpers must stay importable so tests run without a Meep install.

The headline test pins the carrier-averaged, pulse-integrated readout -- the campaign's whole
objective -- against the COMMITTED delay study (chi5_optimization/delay_physics, commit 849274f).
Its run directories are gitignored, so the four raw sub-sample Stokes vectors at tau = 0 are
embedded here verbatim; the expected outputs are the values in that study's own result JSON.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chi5_dbr_design"))

common = pytest.importorskip("common")


# The four pump1-carrier sub-samples at tau = 0, chi0 = 0, from delay_physics/ (SiN
# best_absolute, res 80, decay 1e-4, pad 500 fs, I_pump = 1e12 W/cm^2).
DELAY_PHYSICS_TAU0 = [
    dict(S0=5.065679397510e-02, S1=-4.472861521495e-05, S2=5.049225244993e-02,
         S3=-2.224466245796e-03, dolp=9.967522280661e-01, legacy=1.426904018883e-01),
    dict(S0=5.069836923656e-02, S1=7.969064772821e-06, S2=5.056040825647e-02,
         S3=2.453772520865e-05, dolp=9.972788009922e-01, legacy=1.510024008108e-02),
    dict(S0=5.064596816381e-02, S1=5.237729512128e-05, S2=5.048105895757e-02,
         S3=2.209645409395e-03, dolp=9.967444193517e-01, legacy=-7.709209651676e-02),
    dict(S0=5.060117093843e-02, S1=-2.313138331554e-06, S2=5.046356841693e-02,
         S3=1.735260405361e-05, dolp=9.972806465557e-01, legacy=2.022601104184e-02),
]
# ...and what chi5_optimization/delay_physics reported for them.
EXPECTED = {"theta_chi5_deg": -0.001887, "theta_legacy_deg": 0.025231,
            "theta_fringe_amp_deg": 0.027705, "dolp": 0.997014}


def test_carrier_average_matches_committed_delay_study():
    r = common.carrier_average(DELAY_PHYSICS_TAU0)
    for k, want in EXPECTED.items():
        assert r[k] == pytest.approx(want, abs=1e-6), k


def test_carrier_average_kills_a_pure_fringe():
    """A term with n != m powers of E1 shows up as a sinusoid in the carrier phase, and
    averaging over one period must annihilate it exactly. Build four sub-samples whose S1/S2
    carry a large pure fundamental on top of a small offset; the mean must recover the offset."""
    base = dict(S0=1.0, S3=0.0)
    phi = 2 * np.pi * np.arange(4) / 4
    offset, amp = 1e-4, 1e-2
    recs = [dict(base, S1=offset + amp * np.cos(p), S2=1.0, dolp=1.0, legacy=0.0) for p in phi]
    r = common.carrier_average(recs)
    assert r["S1"] == pytest.approx(offset, abs=1e-12)      # fringe gone
    assert r["vmh_fringe_amp"] == pytest.approx(amp, rel=1e-9)   # and correctly measured


def test_fringe_amplitude_undefined_below_three_samples():
    recs = [dict(S0=1.0, S1=0.1, S2=1.0, S3=0.0, dolp=1.0, legacy=0.0)]
    r = common.carrier_average(recs)
    assert np.isnan(r["theta_fringe_amp_deg"])


def test_pulse_label_gives_a_100fs_intensity_fwhm():
    """The lab pulse is 100 fs FWHM in INTENSITY; the simulator's label is not that quantity.
    intensity FWHM = 2 sqrt(ln2) * T/(2 ln2) = T / sqrt(ln2) = 1.2011 T."""
    assert common.PULSE_LABEL_FS == pytest.approx(83.2555, abs=1e-3)
    intensity_fwhm = common.PULSE_LABEL_FS / np.sqrt(np.log(2.0))
    assert intensity_fwhm == pytest.approx(100.0, abs=1e-6)
    # and the historical label 100.0 really was a 120.1 fs pulse
    assert 100.0 / np.sqrt(np.log(2.0)) == pytest.approx(120.1, abs=0.05)


def test_fwidth_matches_the_simulator_formula():
    c0 = 0.299792458
    for label in (83.2555, 100.0, 150.0):
        want = 1.0 / ((label / (2.0 * np.log(2.0))) * c0)
        assert common.fwidth_of(label) == pytest.approx(want, rel=1e-12)


def test_sidebands_stay_inside_the_probe_readout_band():
    """The pulse-integrated Stokes vector sums the probe DFT band f_probe +- fwidth/2, so a
    Delta that puts the FWM sidebands outside it would silently drop the measured signal."""
    half_band = 0.5 * common.FWIDTH
    assert common.DELTA_MAX_INBAND < half_band
    for d in common.DELTA_GRID:
        assert d <= common.DELTA_MAX_INBAND + 1e-12, d
    assert common.DELTA_RANGE[1] <= half_band


def test_fab_constraints_reject_and_accept_correctly():
    base = {"materials": {"SiN": {}, "SiO2": {}},
            "cavity": {"mat": "SiN", "L_um": 5.0},
            "mirrors": {"left": [{"mat": "SiN", "thk_um": 0.2}, {"mat": "SiO2", "thk_um": 0.3}],
                        "right": [{"mat": "SiO2", "thk_um": 0.3}, {"mat": "SiN", "thk_um": 0.2}]}}
    ok = common.build_geometry(base, 3, 3, 0.24, 0.34, 5.0)
    assert common.fab_violations(ok) == []
    too_thin = common.build_geometry(base, 3, 3, 0.05, 0.34, 5.0)
    assert any("< 0.080" in v for v in common.fab_violations(too_thin))
    too_thick = common.build_geometry(base, 6, 6, 0.55, 0.55, 9.0)
    assert any("stack" in v for v in common.fab_violations(too_thick))
    too_many = common.build_geometry(base, 8, 8, 0.1, 0.1, 2.0)
    assert any("pairs" in v for v in common.fab_violations(too_many))


def test_build_geometry_supports_asymmetric_mirrors():
    """Unequal mirror counts are the campaign's new degree of freedom: a cavity symmetric about
    w_s gives Re[Delta chi] = 0, i.e. zero net rotation, so the symmetry must be breakable."""
    base = {"materials": {"SiN": {}, "SiO2": {}},
            "cavity": {"mat": "SiN", "L_um": 5.0},
            "mirrors": {"left": [{"mat": "SiN", "thk_um": 0.2}, {"mat": "SiO2", "thk_um": 0.3}],
                        "right": [{"mat": "SiO2", "thk_um": 0.3}, {"mat": "SiN", "thk_um": 0.2}]}}
    g = common.build_geometry(base, 5, 2, 0.24, 0.34, 6.0)
    assert len(g["mirrors"]["left"]) == 10 and len(g["mirrors"]["right"]) == 4
    assert g["mirrors"]["left"][0]["mat"] == "SiN"      # ordering preserved from the base
    assert g["mirrors"]["right"][0]["mat"] == "SiO2"
    p = common.geometry_params(g)
    assert (p["n_left"], p["n_right"]) == (5, 2)
    assert p["stack_um"] == pytest.approx(7 * (0.24 + 0.34) + 6.0)


def test_subsample_taus_span_exactly_one_carrier_period():
    f1 = 0.6573                      # 1/um, the baseline pump1
    taus = common.subsample_taus(f1, 4)
    T1 = common.carrier_period_fs(f1)
    assert T1 == pytest.approx(5.075, abs=1e-3)
    assert taus[0] == 0.0
    assert np.allclose(np.diff(taus), T1 / 4)
    assert taus[-1] + T1 / 4 == pytest.approx(T1)


def _have_material_csvs():
    d = HERE.parent
    return (d / "si3n4.csv").exists() and (d / "sio2.csv").exists()


@pytest.mark.skipif(not _have_material_csvs(), reason="ellipsometry CSVs not present")
def test_pump_pairs_recovers_the_fabricated_design_point():
    """The Stage-2 grid originally straddled pumps about a single mode, which cannot express
    best_absolute: it puts each pump on its OWN mode (0.6573 / 0.6353, Delta = 0.0219) with a
    center that is not a mode.  The baseline was therefore denied its own operating point and
    scored below its known value, inflating every 'N x vs baseline' figure.  pump_pairs must
    offer that configuration back."""
    geom = common.load_base_geometry()
    modes = common.load_base_modes()
    c_design = 0.5 * (modes["pump1"]["frequency"] + modes["pump2"]["frequency"])
    d_design = modes["pump1"]["frequency"] - modes["pump2"]["frequency"]
    pairs = common.pump_pairs(geom, max_pairs=4)
    assert pairs, "no resonant pump pairs found for the fabricated design"
    assert any(abs(c - c_design) < 5e-3 and abs(d - d_design) < 2e-3 for c, d in pairs), pairs
    # every offered pair must be inside the readout-band-limited Delta range
    for _c, d in pairs:
        assert common.DELTA_RANGE[0] <= d <= common.DELTA_RANGE[1]


@pytest.mark.skipif(not _have_material_csvs(), reason="ellipsometry CSVs not present")
def test_pump_centers_cover_the_band_not_just_high_Q():
    """Ranking pump centers by Q is meaningless here -- every pump-band mode has Q >> the 100 fs
    Q_cap (~12), so buildup is saturated and Q carries no information.  pump_centers must
    therefore spread across the band rather than clustering on the highest-Q modes."""
    geom = common.load_base_geometry()
    top_q = [m["freq"] for m in common.pump_modes(geom)[:2]]
    centers = [m["freq"] for m in common.pump_centers(geom, n_q=2, n_span=3)]
    assert all(f in centers for f in top_q)          # high-Q ones are kept
    assert len(centers) > len(top_q)                 # ...and the band is also covered
    lo, hi = 1.0 / common.PUMP_BAND[1], 1.0 / common.PUMP_BAND[0]
    # the added centers must reach the upper half of the band, where top-2-by-Q did not
    assert max(centers) > 0.5 * (lo + hi), (centers, lo, hi)
    # every pump-band mode is far above the 100 fs cap, which is why Q cannot rank them
    q_cap = min(centers) / common.FWIDTH
    assert q_cap < 20
    assert all(m["Q"] > 2 * q_cap for m in common.pump_modes(geom))


def test_spearman_against_known_cases():
    assert common.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert common.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert common.spearman([1, 2, 3, 4], [1, 4, 2, 3]) == pytest.approx(0.4, abs=1e-9)
