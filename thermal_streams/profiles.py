"""Analytic stream-temperature profiles along a counter-flow exchanger.

Coordinate
----------
``s in [0, 1]`` is the fractional cumulative heat-transfer area measured
from the hot-stream inlet end: ``s = 0`` is where the hot stream enters
(and the cold stream leaves); ``s = 1`` is where the hot stream leaves
(and the cold stream enters). ``A(s) = s * A_tot``.

Model (identical to the eps-NTU model, see docs/model.md): steady state,
one-dimensional, uniform overall heat-transfer coefficient U, constant
capacity rates, no axial conduction, no heat loss to ambient. With ``a``
the cumulative transfer area measured from the hot inlet,

    dT_h/da = -(U / C_h) (T_h - T_c)
    dT_c/da = -(U / C_c) (T_h - T_c)

(the cold stream flows toward decreasing ``a``). Hence the local
temperature difference decays exponentially,

    dT(s) = dT(0) * exp(-k A_tot s),   k = U (1/C_h - 1/C_c),

which degenerates to a constant difference when C_h = C_c. Integrating the
hot-stream equation gives the profiles below; they reproduce the
eps-NTU outlet temperatures exactly (verified in tests and
docs/verification.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_streams.core import (
    FLOW_COUNTER,
    ExchangerConfig,
    ExchangerResult,
    solve_exchanger,
)

__all__ = ["CounterFlowProfile", "counter_flow_profile"]


@dataclass(frozen=True)
class CounterFlowProfile:
    """Position-resolved temperatures along a counter-flow exchanger."""

    #: Fractional transfer area from the hot inlet, 0 (hot in / cold out)
    #: to 1 (hot out / cold in).
    s: np.ndarray
    #: Hot-stream temperature [K] at each ``s``.
    T_h_K: np.ndarray
    #: Cold-stream temperature [K] at each ``s``.
    T_c_K: np.ndarray
    #: The eps-NTU rating solution the profile is built from.
    result: ExchangerResult


def counter_flow_profile(
    config: ExchangerConfig, n_points: int = 201
) -> CounterFlowProfile:
    """Analytic temperature profiles for a counter-flow exchanger.

    Parameters
    ----------
    config : ExchangerConfig
        A validated exchanger configuration with ``flow="counter"``.
    n_points : int
        Number of equally spaced stations in ``s in [0, 1]`` (>= 2).
    """
    if config.flow != FLOW_COUNTER:
        raise ValueError(
            "the closed-form temperature profile is implemented for counter "
            f"flow only; got flow={config.flow!r}"
        )
    if not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError(f"n_points must be an integer >= 2; got {n_points!r}")

    res = solve_exchanger(config)
    c_h = config.C_h_W_K
    c_c = config.C_c_W_K
    u = config.U_W_m2K
    a_tot = config.A_m2

    s = np.linspace(0.0, 1.0, int(n_points))
    d_t0 = config.T_h_in_K - res.T_c_out_K  # terminal difference at s = 0

    if abs(c_h - c_c) <= 1.0e-12 * max(c_h, c_c):
        # Equal capacity rates: constant terminal difference, linear profiles.
        d_t = np.full_like(s, d_t0)
        t_h = config.T_h_in_K - (u * a_tot * d_t0 / c_h) * s
    else:
        k_a = u * a_tot * (1.0 / c_h - 1.0 / c_c)
        d_t = d_t0 * np.exp(-k_a * s)
        t_h = config.T_h_in_K - (c_c / (c_c - c_h)) * (d_t0 - d_t)
    t_c = t_h - d_t

    return CounterFlowProfile(s=s, T_h_K=t_h, T_c_K=t_c, result=res)
