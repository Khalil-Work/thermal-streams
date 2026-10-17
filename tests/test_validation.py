"""(f) Input validation: clear, specific errors for invalid inputs."""


import pytest

from thermal_streams import (
    FLOWS,
    ExchangerConfig,
    effectiveness,
    log_mean_temperature_difference,
)


def valid_kwargs():
    return dict(
        flow="counter",
        U_W_m2K=800.0,
        A_m2=6.0,
        C_h_W_K=1200.0,
        C_c_W_K=2000.0,
        T_h_in_K=363.15,
        T_c_in_K=298.15,
    )


class TestConfigValidation:
    @pytest.mark.parametrize(
        "field,bad_value",
        [
            ("U_W_m2K", 0.0),
            ("U_W_m2K", -250.0),
            ("U_W_m2K", float("nan")),
            ("U_W_m2K", float("inf")),
            ("A_m2", 0.0),
            ("A_m2", -1.0),
            ("A_m2", float("nan")),
            ("C_h_W_K", 0.0),
            ("C_h_W_K", -5.0),
            ("C_c_W_K", 0.0),
            ("C_c_W_K", -1e-6),
            ("T_h_in_K", float("nan")),
            ("T_c_in_K", 0.0),
            ("T_c_in_K", -273.15),
        ],
    )
    def test_nonpositive_or_nonfinite_numbers_are_rejected(self, field, bad_value):
        kwargs = valid_kwargs()
        kwargs[field] = bad_value
        with pytest.raises(ValueError, match=field):
            ExchangerConfig(**kwargs)

    @pytest.mark.parametrize(
        "t_h_in,t_c_in", [(300.0, 300.0), (300.0, 350.0)]
    )
    def test_inlet_temperatures_must_differ_positive(self, t_h_in, t_c_in):
        kwargs = valid_kwargs()
        kwargs["T_h_in_K"] = t_h_in
        kwargs["T_c_in_K"] = t_c_in
        with pytest.raises(ValueError, match="positive inlet temperature difference"):
            ExchangerConfig(**kwargs)

    @pytest.mark.parametrize("flow", ["diagonal", "", "COUNTER", None, 3])
    def test_unknown_flow_arrangement_is_rejected_with_options(self, flow):
        kwargs = valid_kwargs()
        kwargs["flow"] = flow
        with pytest.raises(ValueError, match="flow must be one of"):
            ExchangerConfig(**kwargs)

    def test_error_message_names_the_offending_parameter(self):
        kwargs = valid_kwargs()
        kwargs["U_W_m2K"] = -1.0
        with pytest.raises(ValueError, match=r"U_W_m2K.*strictly positive"):
            ExchangerConfig(**kwargs)


class TestEffectivenessValidation:
    @pytest.mark.parametrize("ntu", [0.0, -1.0, float("inf"), float("nan")])
    def test_invalid_ntu_is_rejected(self, ntu):
        with pytest.raises(ValueError, match="ntu"):
            effectiveness(ntu, 0.5, "counter")

    @pytest.mark.parametrize("c_r", [-0.1, 1.5, float("inf"), float("nan")])
    def test_invalid_c_r_is_rejected(self, c_r):
        with pytest.raises(ValueError, match="c_r"):
            effectiveness(2.0, c_r, "counter")

    def test_c_r_outside_unit_interval_in_arrays_is_rejected(self):
        with pytest.raises(ValueError, match="c_r"):
            effectiveness(2.0, [0.1, 0.5, 1.2], "counter")

    def test_unknown_flow_is_rejected_with_options(self):
        with pytest.raises(ValueError, match="flow must be one of"):
            effectiveness(2.0, 0.5, "spiral")

    def test_boundary_values_are_accepted(self):
        # c_r endpoints of [0, 1] are valid.
        assert 0.0 < effectiveness(2.0, 0.0, "counter") < 1.0
        assert 0.0 < effectiveness(2.0, 1.0, "counter") < 1.0


class TestLmtdValidation:
    @pytest.mark.parametrize(
        "d1,d2", [(0.0, 10.0), (10.0, 0.0), (-3.0, 4.0), (4.0, -3.0),
                  (float("nan"), 4.0), (float("inf"), 4.0), ("hot", 4.0)]
    )
    def test_invalid_terminal_differences_are_rejected(self, d1, d2):
        with pytest.raises(ValueError, match="delta_T"):
            log_mean_temperature_difference(d1, d2)


def test_documented_flow_constants_are_consistent():
    assert isinstance(FLOWS, tuple)
    assert len(FLOWS) == 5
    assert len(set(FLOWS)) == 5
