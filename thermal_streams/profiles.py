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
hot-stream equation gives the profiles below. A normalized integral of a
decaying exponential is evaluated from the better-conditioned boundary;
the C_h > C_c branch starts at the hot outlet / cold inlet. This avoids
amplifying a rounded-to-zero terminal difference at high NTU. Endpoints
match the eps-NTU rating to floating-point precision (see tests and
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
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError(f"n_points must be an integer >= 2; got {n_points!r}")

    res = solve_exchanger(config)
    c_h = config.C_h_W_K
    c_c = config.C_c_W_K
    s = np.linspace(0.0, 1.0, int(n_points))
    # Evaluate a DECAYING exponential from the well-conditioned boundary.
    # Normalize the integral of the gap; never form a tiny rounded terminal
    # gap and multiply it by an exponentially growing factor.
    b = res.ntu * ((max(c_h, c_c) - min(c_h, c_c)) / max(c_h, c_c))
    coordinate = s if c_h <= c_c else 1.0 - s
    if b == 0.0:
        fraction = coordinate
    else:
        fraction = np.expm1(-b * coordinate) / np.expm1(-b)
    hot_drop = res.duty_W / c_h
    cold_rise = res.duty_W / c_c
    if c_h <= c_c:
        t_h = config.T_h_in_K - hot_drop * fraction
        t_c = res.T_c_out_K - cold_rise * fraction
    else:
        t_h = res.T_h_out_K + hot_drop * fraction
        t_c = config.T_c_in_K + cold_rise * fraction

    return CounterFlowProfile(s=s, T_h_K=t_h, T_c_K=t_c, result=res)
