"""(a) Energy balance: the heat lost by the hot stream equals the heat gained
by the cold stream, exactly to roundoff, across parameter sweeps for both
required configurations and the additional arrangements."""
import pytest

from thermal_streams import FLOWS, ExchangerConfig, solve_exchanger


def build_config(flow, ntu, c_r, hot_is_cmin, c_min=1200.0, d_t=65.0,
                 t_c_in=298.15, u=800.0):
    c_max = c_min / c_r if c_r > 0.0 else 1.0e12
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


CASES = [
    (flow, ntu, c_r, hot_is_cmin)
    for flow in FLOWS
    for ntu in [0.3, 1.0, 3.16, 10.0]
    for c_r in [0.15, 0.5, 0.85, 1.0]
    for hot_is_cmin in [True, False]
]


@pytest.mark.parametrize("flow,ntu,c_r,hot_is_cmin", CASES)
def test_energy_balance_closes_to_roundoff(flow, ntu, c_r, hot_is_cmin):
    cfg = build_config(flow, ntu, c_r, hot_is_cmin)
    res = solve_exchanger(cfg)

    assert res.energy_balance_residual_W == pytest.approx(0.0, abs=1e-9)
    assert res.energy_balance_residual_W / res.duty_W < 1e-13


@pytest.mark.parametrize("flow,ntu,c_r,hot_is_cmin", CASES)
def test_stream_duties_match_reported_duty(flow, ntu, c_r, hot_is_cmin):
    cfg = build_config(flow, ntu, c_r, hot_is_cmin)
    res = solve_exchanger(cfg)

    q_hot = cfg.C_h_W_K * (cfg.T_h_in_K - res.T_h_out_K)
    q_cold = cfg.C_c_W_K * (res.T_c_out_K - cfg.T_c_in_K)
    assert q_hot == pytest.approx(res.duty_W, rel=1e-13)
    assert q_cold == pytest.approx(res.duty_W, rel=1e-13)


@pytest.mark.parametrize("flow,ntu,c_r,hot_is_cmin", CASES)
def test_outlet_temperatures_are_physical(flow, ntu, c_r, hot_is_cmin):
    cfg = build_config(flow, ntu, c_r, hot_is_cmin)
    res = solve_exchanger(cfg)

    # Hot stream cools, cold stream heats, both stay inside the inlet band
    # and the local temperature difference stays positive at both terminals.
    assert cfg.T_h_in_K > res.T_h_out_K
    assert res.T_c_out_K > cfg.T_c_in_K
    assert res.T_h_out_K > cfg.T_c_in_K
    assert res.T_c_out_K < cfg.T_h_in_K
