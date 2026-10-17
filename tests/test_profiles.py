"""Position-resolved counter-flow profiles: endpoint consistency, incremental
energy balance, and an independent RK4 integration of the governing ODEs.

The RK4 test integrates

    dT_h/da = -(U/C_h)(T_h - T_c),   dT_c/da = -(U/C_c)(T_h - T_c)

from the hot inlet with the cold outlet temperature as the second initial
value, and checks that the far-end boundary conditions (hot outlet, cold
inlet) emerge from the integration - a fully independent route to the
closed-form profiles.
"""
import numpy as np
import pytest

from thermal_streams import ExchangerConfig, counter_flow_profile, solve_exchanger


def make_config(c_h=1200.0, c_c=2000.0, ntu=4.0, u=800.0,
                t_h_in=363.15, t_c_in=298.15):
    c_min = min(c_h, c_c)
    return ExchangerConfig(
        flow="counter",
        U_W_m2K=u,
        A_m2=ntu * c_min / u,
        C_h_W_K=c_h,
        C_c_W_K=c_c,
        T_h_in_K=t_h_in,
        T_c_in_K=t_c_in,
    )


@pytest.mark.parametrize(
    "c_h,c_c", [(1200.0, 2000.0), (2000.0, 1200.0), (1500.0, 1500.0)]
)
def test_profile_endpoints_match_the_rating_solution(c_h, c_c):
    cfg = make_config(c_h=c_h, c_c=c_c)
    res = solve_exchanger(cfg)
    prof = counter_flow_profile(cfg, 41)

    assert prof.s[0] == 0.0 and prof.s[-1] == 1.0
    assert prof.T_h_K[0] == pytest.approx(cfg.T_h_in_K, abs=1e-12)
    assert prof.T_c_K[0] == pytest.approx(res.T_c_out_K, abs=1e-12)
    assert prof.T_h_K[-1] == pytest.approx(res.T_h_out_K, abs=1e-12)
    assert prof.T_c_K[-1] == pytest.approx(cfg.T_c_in_K, abs=1e-12)


@pytest.mark.parametrize(
    "c_h,c_c", [(1200.0, 2000.0), (2000.0, 1200.0), (1500.0, 1500.0)]
)
def test_profile_satisfies_the_incremental_energy_balance(c_h, c_c):
    # Heat lost by the hot stream between s = 0 and s equals heat gained by
    # the cold stream over the same stations:
    #   C_h (T_h_in - T_h(s)) = C_c (T_c_out - T_c(s)).
    cfg = make_config(c_h=c_h, c_c=c_c)
    prof = counter_flow_profile(cfg, 101)

    q_hot_side = cfg.C_h_W_K * (cfg.T_h_in_K - prof.T_h_K)
    q_cold_side = cfg.C_c_W_K * (prof.T_c_K[0] - prof.T_c_K)
    assert np.allclose(q_hot_side, q_cold_side, rtol=1e-11, atol=1e-6)


@pytest.mark.parametrize(
    "c_h,c_c", [(1200.0, 2000.0), (2000.0, 1200.0), (1500.0, 1500.0)]
)
def test_profile_agrees_with_rk4_integration_of_the_governing_odes(c_h, c_c):
    cfg = make_config(c_h=c_h, c_c=c_c)
    prof = counter_flow_profile(cfg, 201)
    res = prof.result

    u, a_tot = cfg.U_W_m2K, cfg.A_m2

    def deriv(t_h, t_c):
        d_t = t_h - t_c
        return -u / c_h * d_t, -u / c_c * d_t

    n_steps = 4000
    h = a_tot / n_steps
    t_h, t_c = cfg.T_h_in_K, res.T_c_out_K
    th_path = np.empty(201)
    tc_path = np.empty(201)
    th_path[0], tc_path[0] = t_h, t_c
    for step in range(n_steps):
        k1h, k1c = deriv(t_h, t_c)
        k2h, k2c = deriv(t_h + h / 2 * k1h, t_c + h / 2 * k1c)
        k3h, k3c = deriv(t_h + h / 2 * k2h, t_c + h / 2 * k2c)
        k4h, k4c = deriv(t_h + h * k3h, t_c + h * k3c)
        t_h += h / 6 * (k1h + 2 * k2h + 2 * k3h + k4h)
        t_c += h / 6 * (k1c + 2 * k2c + 2 * k3c + k4c)
        if (step + 1) % 20 == 0:
            i = (step + 1) // 20
            th_path[i], tc_path[i] = t_h, t_c

    assert np.allclose(th_path, prof.T_h_K, rtol=0, atol=1e-9)
    assert np.allclose(tc_path, prof.T_c_K, rtol=0, atol=1e-9)
    # The far-end boundary conditions emerge from the integration itself.
    assert th_path[-1] == pytest.approx(res.T_h_out_K, abs=1e-9)
    assert tc_path[-1] == pytest.approx(cfg.T_c_in_K, abs=1e-9)


@pytest.mark.parametrize(
    "c_h,c_c", [(1200.0, 2000.0), (2000.0, 1200.0)]
)
def test_profile_gap_decays_exponentially_with_area(c_h, c_c):
    cfg = make_config(c_h=c_h, c_c=c_c)
    prof = counter_flow_profile(cfg, 81)
    k_a = cfg.U_W_m2K * cfg.A_m2 * (1.0 / cfg.C_h_W_K - 1.0 / cfg.C_c_W_K)
    gap = prof.T_h_K - prof.T_c_K
    assert np.all(gap > 0.0)
    assert np.allclose(np.log(gap / gap[0]), -k_a * prof.s, rtol=1e-12, atol=1e-13)


def test_profile_streams_stay_monotone_and_bounded():
    cfg = make_config()
    prof = counter_flow_profile(cfg, 121)
    res = prof.result

    assert np.all(np.diff(prof.T_h_K) < 0.0)  # hot cools toward its outlet
    assert np.all(np.diff(prof.T_c_K) < 0.0)  # cold cools toward its inlet
    assert np.all(prof.T_h_K >= res.T_h_out_K - 1e-12)
    assert np.all(prof.T_h_K <= cfg.T_h_in_K + 1e-12)
    assert np.all(prof.T_c_K >= cfg.T_c_in_K - 1e-12)
    assert np.all(prof.T_c_K <= res.T_c_out_K + 1e-12)


def test_profile_rejects_non_counter_arrangements_and_bad_sampling():
    cfg = ExchangerConfig(
        flow="parallel",
        U_W_m2K=800.0,
        A_m2=6.0,
        C_h_W_K=1200.0,
        C_c_W_K=2000.0,
        T_h_in_K=363.15,
        T_c_in_K=298.15,
    )
    with pytest.raises(ValueError, match="counter"):
        counter_flow_profile(cfg)
    with pytest.raises(ValueError, match="n_points"):
        counter_flow_profile(make_config(), n_points=1)
