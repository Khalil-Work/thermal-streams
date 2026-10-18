"""Regressions for high-NTU profiles, near-equal capacities and wide LMTD."""
from decimal import Decimal, localcontext
import math
import numpy as np
import pytest

from thermal_streams import (ExchangerConfig, counter_flow_profile,
                             effectiveness, log_mean_temperature_difference,
                             solve_exchanger)


from examples.stability_reference import decimal_profiles


@pytest.mark.parametrize("ch,cc", [(1200., 2000.), (2000., 1200.), (1500., 1500.),
                                   (1500., 1500.*(1+1e-10)), (1500.*(1+1e-10), 1500.)])
@pytest.mark.parametrize("ntu", [4., 50., 100.])
def test_profiles_against_80_digit_reference_in_both_orientations(ch, cc, ntu):
    cfg = ExchangerConfig("counter", 800, ntu*min(ch,cc)/800, ch, cc, 363.15, 298.15)
    profile = counter_flow_profile(cfg, 101)
    ref_hot, ref_cold = decimal_profiles(cfg, profile.s)
    np.testing.assert_allclose(profile.T_h_K, ref_hot, rtol=0, atol=2e-10)
    np.testing.assert_allclose(profile.T_c_K, ref_cold, rtol=0, atol=2e-10)
    result = profile.result
    np.testing.assert_allclose([profile.T_h_K[0], profile.T_h_K[-1], profile.T_c_K[0], profile.T_c_K[-1]],
                               [cfg.T_h_in_K, result.T_h_out_K, result.T_c_out_K, cfg.T_c_in_K],
                               rtol=0, atol=2e-12)
    assert np.all(np.isfinite(profile.T_h_K)) and np.all(np.isfinite(profile.T_c_K))
    assert np.all(np.diff(profile.T_h_K) <= 2e-12)
    assert np.all(np.diff(profile.T_c_K) <= 2e-12)
    assert np.all(profile.T_h_K >= profile.T_c_K - 2e-12)
    assert np.all(profile.T_h_K <= cfg.T_h_in_K + 2e-12)
    assert np.all(profile.T_c_K >= cfg.T_c_in_K - 2e-12)
    q_hot = ch * (cfg.T_h_in_K-profile.T_h_K)
    q_cold = cc * (profile.T_c_K[0]-profile.T_c_K)
    np.testing.assert_allclose(q_hot, q_cold, rtol=0, atol=3e-7)


@pytest.mark.parametrize("ch,cc", [(1200., 2000.), (2000., 1200.)])
def test_high_ntu_against_rk4_from_the_conditioned_end(ch, cc):
    cfg = ExchangerConfig("counter", 800., 150., ch, cc, 363.15, 298.15)
    profile = counter_flow_profile(cfg, 101)
    # Integrate in the decaying direction; never amplify input roundoff by exp(40).
    backwards = ch > cc
    state = np.array([profile.result.T_h_out_K, cfg.T_c_in_K] if backwards else
                     [cfg.T_h_in_K, profile.result.T_c_out_K])
    ds = (-1 if backwards else 1) / 10000
    def rhs(y):
        return -cfg.U_W_m2K*cfg.A_m2 * (y[0]-y[1]) * np.array([1/ch, 1/cc])
    sampled = [state.copy()]
    for step in range(10000):
        k1 = rhs(state)
        k2 = rhs(state+ds*k1/2)
        k3 = rhs(state+ds*k2/2)
        k4 = rhs(state+ds*k3)
        state = state + ds*(k1+2*k2+2*k3+k4)/6
        if (step+1) % 100 == 0:
            sampled.append(state.copy())
    sampled = np.array(sampled)[::-1] if backwards else np.array(sampled)
    np.testing.assert_allclose(sampled[:,0], profile.T_h_K, rtol=0, atol=2e-8)
    np.testing.assert_allclose(sampled[:,1], profile.T_c_K, rtol=0, atol=2e-8)


@pytest.mark.parametrize("ntu,cr", [(100, 1-1e-10), (1e-8, 1-1e-12), (3, 1-1e-8)])
def test_effectiveness_near_equal_capacities_has_no_tolerance_plateau(ntu, cr):
    with localcontext() as ctx:
        ctx.prec = 80
        n, r = Decimal.from_float(float(ntu)), Decimal.from_float(float(cr))
        e = (-n*(1-r)).exp()
        expected = float((1-e)/(1-r*e))
    assert effectiveness(ntu, cr, "counter") == pytest.approx(expected, rel=5e-15)


@pytest.mark.parametrize("lo,hi", [(1e-300, 1e300), (1e-200, 1e100), (1., 1e300),
                                   (1e308, 1e308), (1e308, 1.0000000000001e308)])
def test_lmtd_is_symmetric_finite_and_accurate_over_wide_scales(lo, hi):
    with localcontext() as ctx:
        ctx.prec = 80
        a, b = Decimal.from_float(lo), Decimal.from_float(hi)
        expected = lo if a == b else float((b-a)/(b.ln()-a.ln()))
    value = log_mean_temperature_difference(lo, hi)
    assert math.isfinite(value) and lo <= value <= hi
    assert value == pytest.approx(expected, rel=3e-14)
    assert value == log_mean_temperature_difference(hi, lo)


def test_config_normalizes_float_convertible_inputs_before_rating():
    cfg = ExchangerConfig("counter", "800", "6", "1200", "2000", "363.15", "298.15")
    assert isinstance(cfg.U_W_m2K, float)
    assert math.isfinite(solve_exchanger(cfg).duty_W)


@pytest.mark.parametrize("u,area,ch,cc,thin,tcin", [
    (1e308, 1e308, 1200, 2000, 363.15, 298.15),
    (1e-300, 1e-300, 1200, 2000, 363.15, 298.15),
    (1e308, 1, 1e308, 1e308, 1e308, 1),
])
def test_nonrepresentable_derived_ratings_raise(u, area, ch, cc, thin, tcin):
    cfg = ExchangerConfig("counter", u, area, ch, cc, thin, tcin)
    with pytest.raises(ValueError, match="UA|derived"):
        solve_exchanger(cfg)
