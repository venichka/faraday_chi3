from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("meep")

from pump_intensity_sweep import _load_trace_bundle_cached, _write_trace_bundle


def test_trace_bundle_roundtrip(tmp_path):
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    f = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=float)
    dft_plus = np.ones((3, 5), dtype=float)
    dft_minus = 2.0 * np.ones((3, 5), dtype=float)
    td_plus = 3.0 * np.ones((3, 5), dtype=float)
    td_minus = 4.0 * np.ones((3, 5), dtype=float)
    theta = np.array([1.0, 2.0, 3.0], dtype=float)
    theta_total = np.array([1.5, 2.5, 3.5], dtype=float)

    fake_result = SimpleNamespace(
        dft_traces=SimpleNamespace(time=t, freqs=f, abs_eplus=dft_plus, abs_eminus=dft_minus),
        time_domain_traces=SimpleNamespace(time=t, freqs=f, abs_eplus=td_plus, abs_eminus=td_minus),
        probe_rotation=SimpleNamespace(theta_deg_rel=theta, theta_total_deg_rel=theta_total),
    )

    out = _write_trace_bundle(fake_result, tmp_path)
    assert out.exists()

    point = {"trace_bundle_path": str(out)}
    bundle = _load_trace_bundle_cached(point)
    assert np.array_equal(bundle["dft_time"], t)
    assert np.array_equal(bundle["dft_freqs"], f)
    assert np.array_equal(bundle["dft_abs_eplus"], dft_plus)
    assert np.array_equal(bundle["dft_abs_eminus"], dft_minus)
    assert np.array_equal(bundle["td_abs_eplus"], td_plus)
    assert np.array_equal(bundle["td_abs_eminus"], td_minus)
    assert np.array_equal(bundle["theta_deg_rel"], theta)
    assert np.array_equal(bundle["theta_total_deg_rel"], theta_total)

    # cached path should return same content without re-read side effects
    bundle2 = _load_trace_bundle_cached(point)
    assert bundle2 is bundle


def test_trace_bundle_total_falls_back_to_forward_when_absent(tmp_path):
    # Old-style result without theta_total_deg_rel: bundle still writes the key,
    # falling back to the forward trace (no crash).
    t = np.array([0.0, 1.0], dtype=float)
    f = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=float)
    arr = np.ones((2, 5), dtype=float)
    theta = np.array([1.0, 2.0], dtype=float)

    fake_result = SimpleNamespace(
        dft_traces=SimpleNamespace(time=t, freqs=f, abs_eplus=arr, abs_eminus=arr),
        time_domain_traces=SimpleNamespace(time=t, freqs=f, abs_eplus=arr, abs_eminus=arr),
        probe_rotation=SimpleNamespace(theta_deg_rel=theta),
    )

    out = _write_trace_bundle(fake_result, tmp_path)
    point = {"trace_bundle_path": str(out)}
    bundle = _load_trace_bundle_cached(point)
    assert np.array_equal(bundle["theta_total_deg_rel"], theta)
