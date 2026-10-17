"""Effectiveness relations: exact closed forms, limits, anchors, monotonicity.

Expected values are recomputed inside each test with plain ``math.exp``
arithmetic (an independent code path from the vectorized NumPy
implementation) or from known analytic limits.
"""
import math

import numpy as np
import pytest

from thermal_streams import (
    FLOWS,
    ExchangerConfig,
    effectiveness,
    solve_exchanger,
)


# ---------------------------------------------------------------------------
# (c) C_r = 0 anchor: eps = 1 - exp(-NTU) for every implemented arrangement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ntu", [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
)
@pytest.mark.parametrize("flow", FLOWS)
def test_cr_zero_anchor_matches_closed_form(flow, ntu):
    anchor = 1.0 - math.exp(-ntu)
    assert effectiveness(ntu, 0.0, flow) == pytest.approx(anchor, rel=1e-14, abs=1e-15)


def test_cr_zero_anchor_survives_a_large_c_max_stream_in_solve():
    # A 1e12 W/K cold side is the numerical idealization of a condensing
    # stream at constant temperature; C_r = 1.2e-9.
    c_min, u, t_c_in, d_t = 1200.0, 800.0, 298.15, 65.0
    for ntu in [0.5, 1.0, 3.0, 7.0]:
        cfg = ExchangerConfig(
            flow="counter",
            U_W_m2K=u,
            A_m2=ntu * c_min / u,
            C_h_W_K=c_min,
            C_c_W_K=1.0e12,
            T_h_in_K=t_c_in + d_t,
            T_c_in_K=t_c_in,
        )
        res = solve_exchanger(cfg)
        assert res.effectiveness == pytest.approx(1.0 - math.exp(-ntu), rel=1e-8)
        # The near-constant-temperature stream warms only by Q / C_max,
        # i.e. by ~1e-7 K with C_max = 1e12 W/K - it stays at its inlet value
        # to that (physically correct) tolerance.
        assert res.T_c_out_K == pytest.approx(t_c_in, abs=1e-7)


# ---------------------------------------------------------------------------
# (b) Exact reference values for the required arrangements
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ntu,c_r",
    [(0.1, 0.0), (0.5, 0.25), (1.0, 0.5), (4.0, 0.6), (10.0, 0.9), (0.05, 1.0), (3.0, 1.0)],
)
def test_counter_flow_reference_values(ntu, c_r):
    if c_r < 1.0:
        expected = (1.0 - math.exp(-ntu * (1.0 - c_r))) / (
            1.0 - c_r * math.exp(-ntu * (1.0 - c_r))
        )
    else:
        expected = ntu / (1.0 + ntu)
    assert effectiveness(ntu, c_r, "counter") == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize(
    "ntu,c_r",
    [(0.1, 0.0), (0.5, 0.25), (1.0, 0.5), (4.0, 0.6), (10.0, 1.0), (0.05, 1.0)],
)
def test_parallel_flow_reference_values(ntu, c_r):
    expected = (1.0 - math.exp(-ntu * (1.0 + c_r))) / (1.0 + c_r)
    assert effectiveness(ntu, c_r, "parallel") == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize(
    "ntu,c_r", [(1.0, 1.0), (2.0, 0.25), (4.0, 0.6), (10.0, 0.9)]
)
def test_shell_and_tube_reference_values(ntu, c_r):
    s = math.sqrt(1.0 + c_r * c_r)
    a = ntu * s
    expected = 2.0 / (1.0 + c_r + s * (1.0 + math.exp(-a)) / (1.0 - math.exp(-a)))
    assert effectiveness(ntu, c_r, "shell_and_tube") == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("ntu,c_r", [(1.0, 0.5), (4.0, 0.6), (8.0, 1.0), (2.0, 0.9)])
def test_crossflow_c_max_mixed_reference_values(ntu, c_r):
    expected = (1.0 - math.exp(-c_r * (1.0 - math.exp(-ntu)))) / c_r
    assert effectiveness(ntu, c_r, "crossflow_c_max_mixed") == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("ntu,c_r", [(1.0, 0.5), (4.0, 0.6), (8.0, 1.0), (2.0, 0.9)])
def test_crossflow_c_min_mixed_reference_values(ntu, c_r):
    expected = 1.0 - math.exp(-(1.0 / c_r) * (1.0 - math.exp(-ntu * c_r)))
    assert effectiveness(ntu, c_r, "crossflow_c_min_mixed") == pytest.approx(expected, rel=1e-13)


# ---------------------------------------------------------------------------
# (b) Limits: parallel-flow eps_max = 1/(1+C_r); counter flow eps -> 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c_r", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_parallel_flow_maximum_effectiveness(c_r):
    eps_max = 1.0 / (1.0 + c_r)
    assert effectiveness(300.0, c_r, "parallel") == pytest.approx(eps_max, rel=1e-12)
    for ntu in [0.5, 2.0, 8.0]:
        assert effectiveness(ntu, c_r, "parallel") < eps_max


@pytest.mark.parametrize("ntu,c_r", [(15.0, 0.05), (25.0, 0.25), (40.0, 0.5)])
def test_counter_flow_effectiveness_approaches_one(ntu, c_r):
    gap = (1.0 - c_r) * math.exp(-ntu * (1.0 - c_r)) / (
        1.0 - c_r * math.exp(-ntu * (1.0 - c_r))
    )
    eps = effectiveness(ntu, c_r, "counter")
    assert 1.0 - eps == pytest.approx(gap, rel=1e-9, abs=1e-15)
    assert eps > 1.0 - 1e-4


def test_counter_flow_effectiveness_saturates_at_one_numerically():
    assert effectiveness(200.0, 0.01, "counter") == pytest.approx(1.0, abs=1e-15)


def test_counter_flow_equal_capacity_branch_and_continuity():
    for ntu in [0.5, 2.0, 8.0]:
        assert effectiveness(ntu, 1.0, "counter") == pytest.approx(
            ntu / (1.0 + ntu), rel=1e-15
        )
    # The general branch blends continuously into the equal-capacity branch.
    eps_at_one = effectiveness(4.0, 1.0, "counter")
    eps_just_below = effectiveness(4.0, 1.0 - 1e-7, "counter")
    assert eps_just_below == pytest.approx(eps_at_one, abs=1e-5)


# ---------------------------------------------------------------------------
# Large-NTU limits of the additional arrangements
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("c_r", [0.1, 0.5, 0.9, 1.0])
def test_shell_and_tube_large_ntu_limit(c_r):
    s = math.sqrt(1.0 + c_r * c_r)
    expected = 2.0 / (1.0 + c_r + s)
    assert effectiveness(500.0, c_r, "shell_and_tube") == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("c_r", [0.1, 0.5, 0.9, 1.0])
def test_crossflow_large_ntu_limits(c_r):
    expected_max_mixed = (1.0 - math.exp(-c_r)) / c_r
    expected_min_mixed = 1.0 - math.exp(-1.0 / c_r)
    assert effectiveness(500.0, c_r, "crossflow_c_max_mixed") == pytest.approx(
        expected_max_mixed, rel=1e-12
    )
    assert effectiveness(500.0, c_r, "crossflow_c_min_mixed") == pytest.approx(
        expected_min_mixed, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Universal bounds, small-NTU expansion, physical ordering, monotonicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flow", FLOWS)
@pytest.mark.parametrize("c_r", [0.0, 0.35, 0.7, 1.0])
def test_effectiveness_between_zero_and_min_of_one_and_ntu(flow, c_r):
    ntu = np.geomspace(1e-3, 50.0, 80)
    eps = effectiveness(ntu, c_r, flow)
    assert np.all(eps > 0.0)
    assert np.all(eps <= 1.0)
    assert np.all(eps <= ntu * (1.0 + 1e-12))


@pytest.mark.parametrize("flow", FLOWS)
@pytest.mark.parametrize("c_r", [0.0, 0.5, 1.0])
def test_small_ntu_series_expansion(flow, c_r):
    # Every arrangement shares the universal leading behaviour
    # eps = NTU [1 - (1 + C_r) NTU / 2 + O(NTU^2)].
    ntu = 1e-3
    expected = ntu - (1.0 + c_r) * ntu * ntu / 2.0
    assert effectiveness(ntu, c_r, flow) == pytest.approx(expected, abs=2e-9)


@pytest.mark.parametrize("flow", FLOWS[2:])
@pytest.mark.parametrize("c_r", [0.1, 0.3, 0.6, 0.9, 1.0])
def test_arrangements_bounded_by_parallel_and_counter(flow, c_r):
    # Counter flow is the thermodynamically optimal arrangement and parallel
    # flow the worst; every other arrangement must lie between them.
    ntu = np.linspace(0.1, 12.0, 60)
    eps_parallel = effectiveness(ntu, c_r, "parallel")
    eps_counter = effectiveness(ntu, c_r, "counter")
    eps = effectiveness(ntu, c_r, flow)
    assert np.all(eps >= eps_parallel - 1e-12)
    assert np.all(eps <= eps_counter + 1e-12)


@pytest.mark.parametrize("flow", FLOWS)
@pytest.mark.parametrize("c_r", [0.0, 0.3, 0.7, 1.0])
def test_effectiveness_strictly_increasing_in_ntu(flow, c_r):
    ntu = np.linspace(0.05, 15.0, 300)
    eps = effectiveness(ntu, c_r, flow)
    assert np.all(np.diff(eps) > 0.0)


@pytest.mark.parametrize("flow", FLOWS)
@pytest.mark.parametrize("c_r", [0.0, 0.3, 0.7, 1.0])
def test_effectiveness_non_decreasing_over_extended_ntu_range(flow, c_r):
    ntu = np.linspace(0.05, 100.0, 400)
    eps = effectiveness(ntu, c_r, flow)
    assert np.all(np.diff(eps) >= -1e-15)


# ---------------------------------------------------------------------------
# Vectorization / scalar ergonomics
# ---------------------------------------------------------------------------

def test_vectorized_grid_matches_scalar_calls():
    ntus = [0.3, 1.7, 6.0]
    c_rs = [0.1, 0.55, 1.0]
    grid = effectiveness(np.array(ntus)[:, None], np.array(c_rs)[None, :], "counter")
    assert grid.shape == (3, 3)
    for i, n in enumerate(ntus):
        for j, c in enumerate(c_rs):
            assert grid[i, j] == effectiveness(n, c, "counter")


def test_scalar_input_returns_python_float():
    value = effectiveness(2.0, 0.5, "parallel")
    assert isinstance(value, float)
    assert 0.0 < value < 1.0
