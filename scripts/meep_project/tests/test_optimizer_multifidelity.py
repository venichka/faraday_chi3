from __future__ import annotations

import math

from optimize_cavity_geometry_mf import probe_window_score, proxy_score_from_selected


def test_probe_window_score_exact_is_centered_and_bounded() -> None:
    eps = 0.02
    s_center = probe_window_score(0.8, "exact", eps)
    s_edge = probe_window_score(0.8 + eps, "exact", eps)
    s_outside = probe_window_score(0.8 + 1.1 * eps, "exact", eps)
    assert math.isclose(s_center, 1.0, rel_tol=1e-12)
    assert math.isclose(s_edge, 0.0, abs_tol=1e-12)
    assert math.isclose(s_outside, 0.0, abs_tol=1e-12)


def test_probe_window_score_band_prefers_band_center() -> None:
    s_center = probe_window_score(0.9, "band", 0.02)
    s_near_edge = probe_window_score(0.85, "band", 0.02)
    assert s_center > s_near_edge
    assert s_center > 0.0


def test_proxy_score_rewards_better_reflectance_and_depth() -> None:
    selected_bad = {
        "probe_um": 0.8,
        "pump1_um": 1.4,
        "pump2_um": 1.6,
        "probe_R": 0.35,
        "pump1_R": 0.35,
        "pump2_R": 0.35,
        "probe_depth": 0.05,
        "pump1_depth": 0.05,
        "pump2_depth": 0.05,
        "probe_Q": 30.0,
        "pump1_Q": 30.0,
        "pump2_Q": 30.0,
        "pump_center_frequency_inv_um": 0.66,
        "pump_detune_frequency_inv_um": 0.08,
    }
    selected_good = dict(selected_bad)
    selected_good.update(
        {
            "probe_R": 0.05,
            "pump1_R": 0.05,
            "pump2_R": 0.05,
            "probe_depth": 0.5,
            "pump1_depth": 0.5,
            "pump2_depth": 0.5,
            "probe_Q": 80.0,
            "pump1_Q": 80.0,
            "pump2_Q": 80.0,
        }
    )
    score_bad = proxy_score_from_selected(selected_bad, profile="exact", probe_epsilon=0.02)
    score_good = proxy_score_from_selected(selected_good, profile="exact", probe_epsilon=0.02)
    assert score_good > score_bad
