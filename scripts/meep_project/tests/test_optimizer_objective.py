import numpy as np
import pytest

pytest.importorskip("meep")

from optimize_cavity_geometry import Candidate, _objective_from_summary_data, candidate_score


def _summary_template(theta=5.0, wrapped=5.0, dolp=0.95, std_deg=1.2, s0_rel=0.8):
    return {
        "probe_rotation_deg": {
            "final_relative_deg": float(theta),
            "wrapped_final_relative_deg": float(wrapped),
        },
        "probe_stokes_dft": {
            "tail_weighted": {
                "dolp": float(dolp),
                "theta_relative_std_deg": float(std_deg),
                "S0_rel_max": float(s0_rel),
            }
        },
    }


def test_objective_abs_rotation_mode():
    data = _summary_template(theta=7.5, wrapped=7.5)
    rot, abs_rot, score, details = _objective_from_summary_data(
        data,
        objective_metric="abs_rotation",
        quality_std_ref_deg=15.0,
    )
    assert rot == pytest.approx(7.5)
    assert abs_rot == pytest.approx(7.5)
    assert score == pytest.approx(7.5)
    assert 0.0 <= details["quality_factor"] <= 1.0


def test_objective_quality_weighted_mode_reduces_score():
    data = _summary_template(theta=10.0, wrapped=10.0, dolp=0.6, std_deg=8.0, s0_rel=0.3)
    _, abs_rot, score, details = _objective_from_summary_data(
        data,
        objective_metric="quality_weighted_abs_rotation",
        quality_std_ref_deg=10.0,
    )
    assert score <= abs_rot + 1e-12
    assert score == pytest.approx(abs_rot * details["quality_factor"])


def test_rotation_guard_uses_wrapped_if_raw_is_unphysical():
    data = _summary_template(theta=250.0, wrapped=12.0)
    rot, abs_rot, score, _ = _objective_from_summary_data(
        data,
        objective_metric="abs_rotation",
        quality_std_ref_deg=15.0,
    )
    assert rot == pytest.approx(12.0)
    assert abs_rot == pytest.approx(12.0)
    assert score == pytest.approx(12.0)


def test_candidate_score_fallbacks():
    c = Candidate(
        profile="exact",
        N_per=3,
        t_sin_um=0.2,
        t_sio2_um=0.3,
        L_cav_um=1.8,
        pump1_um=1.5,
        pump2_um=1.6,
        probe_um=0.9,
        probe_reflectance=0.2,
        pump1_reflectance=0.2,
        pump2_reflectance=0.2,
        rotation_deg=4.0,
        abs_rotation_deg=4.0,
        objective_summary="x",
    )
    assert candidate_score(c) == pytest.approx(4.0)

    c.score = 2.5
    assert candidate_score(c) == pytest.approx(2.5)
