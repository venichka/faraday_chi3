import numpy as np
import pytest

pytest.importorskip("meep")

import faraday_meep_fp_circ as fm


def test_intensity_conversion_roundtrip():
    intensity = 1.0e9
    n_lin = 1.7
    amp = fm.intensity_to_meep_amplitude(intensity, n_lin=n_lin)
    recon = float(fm.meep_field_to_intensity(np.array([amp]), n_lin=n_lin)[0])
    assert np.isfinite(recon)
    assert recon == pytest.approx(intensity, rel=1e-9)


def test_wrap_linear_polarization_branch():
    vals = np.array([-270.0, -95.0, -90.0, -45.0, 0.0, 89.9, 90.0, 270.0])
    wrapped = fm.wrap_linear_polarization_deg(vals)
    assert np.all(wrapped >= -90.0)
    assert np.all(wrapped < 90.0)


def test_weighted_linear_mean_deg_simple_case():
    th = np.array([10.0, 12.0, 14.0])
    w = np.array([1.0, 2.0, 1.0])
    mean = fm.weighted_linear_mean_deg(th, w)
    assert np.isfinite(mean)
    assert mean == pytest.approx(12.0, abs=1e-6)
