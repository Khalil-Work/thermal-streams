"""Thermal Streams: heat-exchanger effectiveness and heat-balance analysis.

Original NumPy implementations of the standard closed-form eps-NTU
effectiveness relations (parallel flow, counter flow, one-shell-pass
shell-and-tube, and single-pass crossflow with exactly one fluid mixed),
an energy-balance rating solver, the logarithmic mean temperature
difference, and analytic counter-flow temperature profiles.

See ``docs/model.md`` for every implemented relation with its validity
domain and ``docs/verification.md`` for what the test suite establishes.
"""
from thermal_streams.core import (
    FLOWS,
    FLOW_CROSSFLOW_C_MAX_MIXED,
    FLOW_CROSSFLOW_C_MIN_MIXED,
    FLOW_COUNTER,
    FLOW_PARALLEL,
    FLOW_SHELL_AND_TUBE,
    ExchangerConfig,
    ExchangerResult,
    effectiveness,
    lmtd_duty_W,
    log_mean_temperature_difference,
    solve_exchanger,
    terminal_lmtd_K,
)
from thermal_streams.profiles import CounterFlowProfile, counter_flow_profile

__version__ = "1.0.0"

__all__ = [
    "FLOWS",
    "FLOW_PARALLEL",
    "FLOW_COUNTER",
    "FLOW_SHELL_AND_TUBE",
    "FLOW_CROSSFLOW_C_MAX_MIXED",
    "FLOW_CROSSFLOW_C_MIN_MIXED",
    "ExchangerConfig",
    "ExchangerResult",
    "CounterFlowProfile",
    "effectiveness",
    "solve_exchanger",
    "log_mean_temperature_difference",
    "terminal_lmtd_K",
    "lmtd_duty_W",
    "counter_flow_profile",
    "__version__",
]
