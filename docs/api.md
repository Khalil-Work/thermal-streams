# API

Package `thermal_streams` (flat layout; import works from the repository
root, `pytest` picks it up through the root `conftest.py`). Everything is
pure NumPy; figures/animation live only in `examples/run_verification.py`
so the test suite has no plotting dependency.

## Constants

```python
FLOWS  # tuple of the five supported arrangement names:
FLOW_PARALLEL               # "parallel"
FLOW_COUNTER                # "counter"
FLOW_SHELL_AND_TUBE         # "shell_and_tube" (one shell pass, 2+ tube passes)
FLOW_CROSSFLOW_C_MAX_MIXED  # "crossflow_c_max_mixed"
FLOW_CROSSFLOW_C_MIN_MIXED  # "crossflow_c_min_mixed"
```

## `effectiveness(ntu, c_r, flow=FLOW_COUNTER)`

Effectiveness `eps = f(NTU, C_r)` for one arrangement (exact relations in
[docs/model.md](model.md)). Accepts scalars or array-likes (broadcast
against each other); scalar inputs return a Python `float`. Raises
`ValueError` with a message naming the offending value if `ntu` is not
finite and strictly positive, if `c_r` is outside `[0, 1]`, or if `flow`
is unknown.

```python
>>> effectiveness(4.0, 0.6, "counter")
0.9081100343387443
```

## `ExchangerConfig` (frozen dataclass)

```python
ExchangerConfig(
    flow: str,        # one of FLOWS
    U_W_m2K: float,   # overall heat-transfer coefficient, > 0
    A_m2: float,      # heat-transfer area, > 0
    C_h_W_K: float,   # hot-stream capacity rate m_dot*c_p, > 0
    C_c_W_K: float,   # cold-stream capacity rate m_dot*c_p, > 0
    T_h_in_K: float,  # hot inlet temperature, K, > 0
    T_c_in_K: float,  # cold inlet temperature, K, > 0, and < T_h_in_K
)
```

`__post_init__` validates everything: non-finite or non-positive numbers
are rejected with a `ValueError` naming the parameter, and the inlet
temperatures must satisfy `T_h_in_K > T_c_in_K` (positive inlet
temperature difference). Float-convertible inputs are stored as normalized
Python floats rather than leaving strings inside the frozen dataclass.

## `solve_exchanger(config) -> ExchangerResult`

Solves the rating problem: computes `NTU`, `C_r`, `eps`, then

```
Q       = eps * C_min * (T_h_in - T_c_in)
T_h_out = T_h_in - Q / C_h
T_c_out = T_c_in + Q / C_c
```

Derived UA, NTU and duty must also be finite and strictly positive;
unrepresentable overflow/underflow combinations raise `ValueError`.

`ExchangerResult` (frozen dataclass) fields:

| Field | Meaning |
|---|---|
| `config` | the input configuration |
| `ntu` | U A / C_min |
| `c_r` | C_min / C_max |
| `c_min_W_K`, `c_max_W_K` | capacity rates |
| `effectiveness` | eps |
| `duty_W` | Q [W] |
| `T_h_out_K`, `T_c_out_K` | outlet temperatures [K] |
| `energy_balance_residual_W` | property: abs(Q lost by hot - Q gained by cold) [W]; zero to roundoff |

## `log_mean_temperature_difference(delta_T1_K, delta_T2_K)`

`(dT1 - dT2) / ln(dT1 / dT2)` with the equal-difference limit, evaluated
via `log1p` near `dT1 = dT2` and log differences for wide ratios;
the near-equal mean avoids overflow. Both arguments must be finite
and positive (`ValueError` otherwise).

## `terminal_lmtd_K(result)` and `lmtd_duty_W(result)`

LMTD formed from the terminal differences of a solved case and the duty
`Q = U A dT_lm`. Supported for parallel and counter flow only; for the
other arrangements these raise `NotImplementedError` because the LMTD
method needs a geometry correction factor F there (see
[docs/model.md](model.md)).

## `counter_flow_profile(config, n_points=201) -> CounterFlowProfile`

Analytic position-resolved temperatures for `flow="counter"` (derivation
in [docs/model.md](model.md)). `CounterFlowProfile` fields: `s` (fractional
transfer area from the hot inlet, 0 = hot in / cold out, 1 = hot out /
cold in), `T_h_K`, `T_c_K` (arrays, length `n_points`), `result` (the
`ExchangerResult` the profile reproduces). Raises `ValueError` for other
arrangements, boolean/noninteger sampling or `n_points < 2`. The profile
evaluates a decaying exponential from the well-conditioned boundary for
either capacity orientation; see the normalized-integral equations in the
model guide. High-NTU fields do not depend on a tiny subtracted inlet gap.

## Typical session

```python
from thermal_streams import ExchangerConfig, solve_exchanger

cfg = ExchangerConfig(flow="counter", U_W_m2K=800.0, A_m2=6.0,
                      C_h_W_K=1200.0, C_c_W_K=2000.0,
                      T_h_in_K=363.15, T_c_in_K=298.15)
res = solve_exchanger(cfg)
print(f"NTU        = {res.ntu:.3f}")
print(f"C_r        = {res.c_r:.3f}")
print(f"effectiveness = {res.effectiveness:.6f}")
print(f"duty       = {res.duty_W:.1f} W")
print(f"T_h_out    = {res.T_h_out_K - 273.15:.2f} degC")
print(f"T_c_out    = {res.T_c_out_K - 273.15:.2f} degC")
print(f"energy-balance residual = {res.energy_balance_residual_W:.2e} W")
```

Output (executed, Python 3.13.5 / numpy 2.1.3):

```
NTU        = 4.000
C_r        = 0.600
effectiveness = 0.908110
duty       = 70832.6 W
T_h_out    = 30.97 degC
T_c_out    = 60.42 degC
energy-balance residual = 2.91e-11 W
```
