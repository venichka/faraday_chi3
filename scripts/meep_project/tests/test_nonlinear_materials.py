from nonlinear_materials import (
    canonical_high_index_material,
    get_high_index_preset,
    n2_to_chi3_si,
    resolve_high_index_index,
    resolve_high_index_kappa,
    resolve_high_index_n2,
)


def test_high_index_aliases_and_defaults():
    assert canonical_high_index_material("Si3N4") == "sin"
    assert canonical_high_index_material("titania") == "tio2"

    sin = get_high_index_preset("sin")
    tio2 = get_high_index_preset("tio2")
    assert sin.n_const > 0.0
    assert tio2.n_const > sin.n_const

    assert resolve_high_index_index(None, "sin") == sin.n_const
    assert resolve_high_index_kappa(None, "tio2") == tio2.k_const
    assert resolve_high_index_n2(None, "tio2") == tio2.n2_m2_per_w


def test_n2_to_chi3_scales_with_n2():
    nlin = 2.0
    chi3_lo = n2_to_chi3_si(1e-19, nlin)
    chi3_hi = n2_to_chi3_si(5e-19, nlin)
    assert chi3_lo > 0.0
    assert chi3_hi > chi3_lo
