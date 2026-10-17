"""(d) LMTD cross-check: the duty from the eps-NTU method equals the duty
from the independent mean-temperature-difference formulation
Q = U A dT_lm, computed from the outlet temperatures of the same solution.

This is the strongest check in the suite: the two duties are produced by
completely different mathematical routes (closed-form effectiveness vs.
logarithmic mean of the terminal temperature differences).
"""
import math

import numpy as np
import pytest

from thermal_streams import (
    ExchangerConfig,
    log_mean_temperature_difference,
    lmtd_duty_W,
    solve_exchanger,
)


def build_config(flow, ntu, c_r, hot_is_cmin, c_min=1200.0, d_t=65.0,
                 t_c_in=298.15, u=800.0):
    c_max = c_min / c_r
    c_h, c_c = (c_min, c_max) if hot_is_cmin else (c_max, c_min)
    return ExchangerConfig(
        flow=flow,
        U_W_m2K=u,
        A_m2=ntu * c_min / u,
        C_h_W_K=c_h,
        C_c_W_K=c_c,
        T_h_in_K=t_c_in + d_t,
        T_c_in_K=t_c_in,
    )


@pytest.mark.parametrize("c_r", [0.2, 0.5, 0.8, 0.95, 0.99, 1.0])
@pytest.mark.parametrize("hot_is_cmin", [True, False])
def test_counter_flow_duty_agrees_with_lmtd(c_r, hot_is_cmin):
    ntu_grid = np.geomspace(0.1, 15.8, 24)
    for ntu in ntu_grid:
        res = solve_exchanger(build_config("counter", ntu, c_r, hot_is_cmin))
        assert lmtd_duty_W(res) == pytest.approx(res.duty_W, rel=1e-10)


@pytest.mark.parametrize("c_r", [0.2, 0.5, 0.8, 0.95, 0.99, 1.0])
@pytest.mark.parametrize("hot_is_cmin", [True, False])
def test_parallel_flow_duty_agrees_with_lmtd(c_r, hot_is_cmin):
    # The identity is exact analytically. Numerically the outlet temperature
    # difference of a parallel-flow exchanger collapses toward
    # dT_in * exp(-NTU (1 + C_r)); once it approaches the ~1e-14 K roundoff
    # floor of the ~300 K outlet temperatures the LMTD evaluation loses
    # digits. The sweep is therefore restricted to NTU <= 5, where the
    # agreement still holds to 1e-10 relative.
    ntu_grid = np.geomspace(0.1, 5.0, 20)
    for ntu in ntu_grid:
        res = solve_exchanger(build_config("parallel", ntu, c_r, hot_is_cmin))
        assert lmtd_duty_W(res) == pytest.approx(res.duty_W, rel=1e-10)


def test_counter_flow_equal_capacity_rates_is_the_degenerate_limit():
    # C_r = 1 exactly: both terminal differences are equal and the LMTD
    # degenerates to that common value; the closed-form check still holds.
    res = solve_exchanger(build_config("counter", 4.0, 1.0, True))
    d_t = res.config.T_h_in_K - res.T_c_out_K
    assert res.T_h_out_K - res.config.T_c_in_K == pytest.approx(d_t, rel=1e-12)
    assert lmtd_duty_W(res) == pytest.approx(res.duty_W, rel=1e-10)


def test_lmtd_equal_differences_return_the_common_value():
    assert log_mean_temperature_difference(50.0, 50.0) == pytest.approx(50.0)
    nearly_equal = 50.0 * (1.0 + 1e-13)
    assert log_mean_temperature_difference(
        nearly_equal, 50.0
    ) == pytest.approx(50.0, rel=1e-10)


def test_lmtd_known_value_for_a_two_to_one_ratio():
    assert log_mean_temperature_difference(2.0, 1.0) == pytest.approx(
        1.0 / math.log(2.0), rel=1e-14
    )
    # Symmetry in the argument order.
    assert log_mean_temperature_difference(1.0, 2.0) == pytest.approx(
        log_mean_temperature_difference(2.0, 1.0), rel=1e-14
    )


@pytest.mark.parametrize(
    "d1,d2", [(0.0, 5.0), (5.0, 0.0), (-1.0, 5.0), (5.0, -1.0),
              (float("nan"), 5.0), (5.0, float("inf"))]
)
def test_lmtd_requires_positive_finite_arguments(d1, d2):
    with pytest.raises(ValueError):
        log_mean_temperature_difference(d1, d2)


@pytest.mark.parametrize("flow", ["shell_and_tube", "crossflow_c_max_mixed",
                                  "crossflow_c_min_mixed"])
def test_lmtd_method_rejected_for_arrangements_needing_an_f_factor(flow):
    cfg = build_config(flow, 3.0, 0.6, True)
    res = solve_exchanger(cfg)
    with pytest.raises(NotImplementedError):
        lmtd_duty_W(res)
