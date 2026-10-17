"""Core effectiveness (eps-NTU) relations and heat-balance solver.

Original implementations of the standard closed-form effectiveness equations
for heat exchangers. The exact formulas, their derivations and validity
domains are written out in ``docs/model.md``.

Conventions
-----------
- Capacity rate ``C = m_dot * c_p`` [W/K] for each stream.
- ``C_min = min(C_h, C_c)``, ``C_max = max(C_h, C_c)``,
  capacity-rate ratio ``C_r = C_min / C_max`` in ``[0, 1]``.
- Number of transfer units ``NTU = U * A / C_min``.
- Effectiveness ``eps = Q / (C_min * (T_h_in - T_c_in))``.

Units are SI throughout: U in W/(m^2 K), A in m^2, C in W/K, T in K,
duty Q in W.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "FLOW_PARALLEL",
    "FLOW_COUNTER",
    "FLOW_SHELL_AND_TUBE",
    "FLOW_CROSSFLOW_C_MAX_MIXED",
    "FLOW_CROSSFLOW_C_MIN_MIXED",
    "FLOWS",
    "ExchangerConfig",
    "ExchangerResult",
    "effectiveness",
    "solve_exchanger",
    "log_mean_temperature_difference",
    "terminal_lmtd_K",
    "lmtd_duty_W",
]

#: Parallel (co-current) flow, both streams entering at the same end.
FLOW_PARALLEL = "parallel"
#: Counter (counter-current) flow.
FLOW_COUNTER = "counter"
#: One shell pass, 2, 4, ... tube passes (TEMA 1-2 type).
FLOW_SHELL_AND_TUBE = "shell_and_tube"
#: Single-pass crossflow, C_max perfectly mixed across its flow cross-section,
#: C_min unmixed (in separated passageways).
FLOW_CROSSFLOW_C_MAX_MIXED = "crossflow_c_max_mixed"
#: Single-pass crossflow, C_min perfectly mixed, C_max unmixed.
FLOW_CROSSFLOW_C_MIN_MIXED = "crossflow_c_min_mixed"

FLOWS = (
    FLOW_PARALLEL,
    FLOW_COUNTER,
    FLOW_SHELL_AND_TUBE,
    FLOW_CROSSFLOW_C_MAX_MIXED,
    FLOW_CROSSFLOW_C_MIN_MIXED,
)

#: |C_r - 1| at or below this the equal-capacity-rate closed form is used
#: for counter flow (the general expression is a 0/0 limit form there).
_CR_EQUALITY_TOL = 1.0e-9
#: C_r at or below this the C_r = 0 (phase-change) anchor is used for the
#: crossflow one-fluid-mixed relations (also 0/0 limit forms there).
_CR_ZERO_TOL = 1.0e-15


def _require_positive_finite(name: str, value) -> float:
    """Validate that *value* is a finite, strictly positive real number."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number; got {value!r}") from None
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(
            f"{name} must be finite and strictly positive; got {value!r}"
        )
    return v


# ---------------------------------------------------------------------------
# Effectiveness relations eps = f(NTU, C_r)
# ---------------------------------------------------------------------------

def _eps_parallel(ntu: np.ndarray, c_r: np.ndarray) -> np.ndarray:
    """eps = [1 - exp(-NTU (1 + C_r))] / (1 + C_r)."""
    return -np.expm1(-ntu * (1.0 + c_r)) / (1.0 + c_r)


def _eps_counter(ntu: np.ndarray, c_r: np.ndarray) -> np.ndarray:
    """eps = [1 - exp(-NTU (1 - C_r))] / [1 - C_r exp(-NTU (1 - C_r))], C_r < 1;
    eps = NTU / (1 + NTU) at C_r = 1."""
    x = ntu * (1.0 - c_r)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        general = -np.expm1(-x) / (1.0 - c_r * np.exp(-x))
    equal_capacity = ntu / (1.0 + ntu)
    return np.where(np.abs(c_r - 1.0) <= _CR_EQUALITY_TOL, equal_capacity, general)


def _eps_shell_and_tube(ntu: np.ndarray, c_r: np.ndarray) -> np.ndarray:
    """eps = 2 / [1 + C_r + sqrt(1 + C_r^2) (1 + e^-a)/(1 - e^-a)],
    a = NTU sqrt(1 + C_r^2); one shell pass, 2, 4, ... tube passes."""
    a = ntu * np.sqrt(1.0 + c_r * c_r)
    with np.errstate(divide="ignore", over="ignore"):
        ratio = 1.0 / np.tanh(a / 2.0)  # == (1 + exp(-a)) / (1 - exp(-a))
    return 2.0 / (1.0 + c_r + np.sqrt(1.0 + c_r * c_r) * ratio)


def _eps_crossflow_c_max_mixed(ntu: np.ndarray, c_r: np.ndarray) -> np.ndarray:
    """eps = (1/C_r) [1 - exp(-C_r (1 - exp(-NTU)))], C_max mixed."""
    anchor = -np.expm1(-ntu)
    inner = anchor  # 1 - exp(-NTU)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        safe_c_r = np.where(c_r <= _CR_ZERO_TOL, 1.0, c_r)
        out = -np.expm1(-safe_c_r * inner) / safe_c_r
    return np.where(c_r <= _CR_ZERO_TOL, anchor, out)


def _eps_crossflow_c_min_mixed(ntu: np.ndarray, c_r: np.ndarray) -> np.ndarray:
    """eps = 1 - exp[-(1/C_r) (1 - exp(-NTU C_r))], C_min mixed."""
    anchor = -np.expm1(-ntu)
    inner = -np.expm1(-ntu * c_r)  # 1 - exp(-NTU C_r)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        safe_c_r = np.where(c_r <= _CR_ZERO_TOL, 1.0, c_r)
        out = -np.expm1(-inner / safe_c_r)
    return np.where(c_r <= _CR_ZERO_TOL, anchor, out)


_EPS_FNS = {
    FLOW_PARALLEL: _eps_parallel,
    FLOW_COUNTER: _eps_counter,
    FLOW_SHELL_AND_TUBE: _eps_shell_and_tube,
    FLOW_CROSSFLOW_C_MAX_MIXED: _eps_crossflow_c_max_mixed,
    FLOW_CROSSFLOW_C_MIN_MIXED: _eps_crossflow_c_min_mixed,
}


def effectiveness(ntu, c_r, flow=FLOW_COUNTER):
    """Effectiveness ``eps = f(NTU, C_r)`` for a supported arrangement.

    Parameters
    ----------
    ntu : float or array_like
        Number of transfer units, NTU = U A / C_min; finite and > 0.
    c_r : float or array_like
        Capacity-rate ratio C_r = C_min / C_max; must lie in [0, 1].
        C_r = 0 is the condenser/evaporator idealization (one stream at
        effectively constant temperature).
    flow : str
        One of :data:`FLOWS`.

    Returns
    -------
    float or numpy.ndarray
        Effectiveness, broadcast over the inputs. Scalar inputs return a
        Python float.
    """
    if flow not in _EPS_FNS:
        raise ValueError(
            f"flow must be one of {FLOWS}; got {flow!r}"
        )
    ntu_arr = np.asarray(ntu, dtype=float)
    c_r_arr = np.asarray(c_r, dtype=float)

    bad_ntu = ~np.isfinite(ntu_arr) | (ntu_arr <= 0.0)
    if bad_ntu.any():
        idx = int(np.argmax(bad_ntu))
        raise ValueError(
            "ntu must be finite and strictly positive; "
            f"got ntu[{idx}] = {ntu_arr.flat[idx]!r}"
        )
    bad_c_r = ~np.isfinite(c_r_arr) | (c_r_arr < 0.0) | (c_r_arr > 1.0)
    if bad_c_r.any():
        idx = int(np.argmax(bad_c_r))
        raise ValueError(
            "c_r must lie in [0, 1] (= C_min / C_max); "
            f"got c_r[{idx}] = {c_r_arr.flat[idx]!r}"
        )

    eps = _EPS_FNS[flow](ntu_arr, c_r_arr)
    if eps.ndim == 0:
        return float(eps)
    return eps


# ---------------------------------------------------------------------------
# Rating problem: eps-NTU -> duty and outlet temperatures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExchangerConfig:
    """Steady-state rating inputs for one heat exchanger.

    Attributes
    ----------
    flow : str
        Flow arrangement; one of :data:`FLOWS`.
    U_W_m2K : float
        Overall heat-transfer coefficient [W/(m^2 K)], > 0.
    A_m2 : float
        Total heat-transfer surface area [m^2], > 0.
    C_h_W_K, C_c_W_K : float
        Capacity rates of the hot and cold streams [W/K], > 0.
    T_h_in_K, T_c_in_K : float
        Hot- and cold-side inlet temperatures [K]; T_h_in_K > T_c_in_K.
    """

    flow: str
    U_W_m2K: float
    A_m2: float
    C_h_W_K: float
    C_c_W_K: float
    T_h_in_K: float
    T_c_in_K: float

    def __post_init__(self) -> None:
        if self.flow not in FLOWS:
            raise ValueError(f"flow must be one of {FLOWS}; got {self.flow!r}")
        _require_positive_finite("U_W_m2K", self.U_W_m2K)
        _require_positive_finite("A_m2", self.A_m2)
        _require_positive_finite("C_h_W_K", self.C_h_W_K)
        _require_positive_finite("C_c_W_K", self.C_c_W_K)
        t_h = _require_positive_finite("T_h_in_K", self.T_h_in_K)
        t_c = _require_positive_finite("T_c_in_K", self.T_c_in_K)
        if t_h <= t_c:
            raise ValueError(
                "hot-inlet temperature must exceed cold-inlet temperature "
                "(positive inlet temperature difference required); got "
                f"T_h_in_K={self.T_h_in_K!r}, T_c_in_K={self.T_c_in_K!r}"
            )


@dataclass(frozen=True)
class ExchangerResult:
    """Solution of the rating problem for one :class:`ExchangerConfig`."""

    config: ExchangerConfig
    ntu: float
    c_r: float
    c_min_W_K: float
    c_max_W_K: float
    effectiveness: float
    duty_W: float
    T_h_out_K: float
    T_c_out_K: float

    @property
    def energy_balance_residual_W(self) -> float:
        """|Q lost by hot stream - Q gained by cold stream| [W].

        Zero to roundoff by construction; it quantifies the internal
        consistency of the outlet-temperature computation.
        """
        q_hot = self.config.C_h_W_K * (self.config.T_h_in_K - self.T_h_out_K)
        q_cold = self.config.C_c_W_K * (self.T_c_out_K - self.config.T_c_in_K)
        return abs(q_hot - q_cold)


def solve_exchanger(config: ExchangerConfig) -> ExchangerResult:
    """Solve the rating problem: eps-NTU -> duty and both outlet temperatures.

    Q = eps * C_min * (T_h_in - T_c_in); the outlet temperatures follow from
    energy balances on each stream:
    T_h_out = T_h_in - Q / C_h and T_c_out = T_c_in + Q / C_c.
    """
    c_min = min(config.C_h_W_K, config.C_c_W_K)
    c_max = max(config.C_h_W_K, config.C_c_W_K)
    ntu = config.U_W_m2K * config.A_m2 / c_min
    c_r = c_min / c_max
    eps = effectiveness(ntu, c_r, config.flow)
    d_t_in = config.T_h_in_K - config.T_c_in_K
    duty = eps * c_min * d_t_in
    t_h_out = config.T_h_in_K - duty / config.C_h_W_K
    t_c_out = config.T_c_in_K + duty / config.C_c_W_K
    return ExchangerResult(
        config=config,
        ntu=ntu,
        c_r=c_r,
        c_min_W_K=c_min,
        c_max_W_K=c_max,
        effectiveness=eps,
        duty_W=duty,
        T_h_out_K=t_h_out,
        T_c_out_K=t_c_out,
    )


# ---------------------------------------------------------------------------
# Logarithmic mean temperature difference (LMTD)
# ---------------------------------------------------------------------------

def log_mean_temperature_difference(delta_T1_K, delta_T2_K) -> float:
    """LMTD = (dT1 - dT2) / ln(dT1 / dT2), with the dT1 == dT2 limit.

    Uses ``log1p`` so the expression stays accurate when the two terminal
    differences are nearly equal. Both arguments must be finite and > 0.
    """
    d1 = _require_positive_finite("delta_T1_K", delta_T1_K)
    d2 = _require_positive_finite("delta_T2_K", delta_T2_K)
    diff = d1 - d2
    if abs(diff) <= 1.0e-12 * max(d1, d2):
        return 0.5 * (d1 + d2)
    return diff / math.log1p(diff / d2)


def terminal_lmtd_K(result: ExchangerResult) -> float:
    """LMTD formed from the terminal temperature differences of a solution.

    Counter flow: dT1 = T_h_in - T_c_out, dT2 = T_h_out - T_c_in.
    Parallel flow: dT1 = T_h_in - T_c_in, dT2 = T_h_out - T_c_out.

    Raises
    ------
    NotImplementedError
        For arrangements whose LMTD method requires a correction factor F
        (shell-and-tube, crossflow); only the F = 1 arrangements
        (parallel, counter) are supported.
    """
    cfg = result.config
    if cfg.flow == FLOW_COUNTER:
        d1 = cfg.T_h_in_K - result.T_c_out_K
        d2 = result.T_h_out_K - cfg.T_c_in_K
    elif cfg.flow == FLOW_PARALLEL:
        d1 = cfg.T_h_in_K - cfg.T_c_in_K
        d2 = result.T_h_out_K - result.T_c_out_K
    else:
        raise NotImplementedError(
            "the LMTD method without a correction factor F is only exact for "
            f"parallel and counter flow; got flow={cfg.flow!r}"
        )
    return log_mean_temperature_difference(d1, d2)


def lmtd_duty_W(result: ExchangerResult) -> float:
    """Duty from the independent LMTD formulation: Q = U A dT_lm [W]."""
    cfg = result.config
    return cfg.U_W_m2K * cfg.A_m2 * terminal_lmtd_K(result)
