import numpy as np
import pytest

from gEconpy.model.perfect_foresight.validation import validate_perfect_foresight_inputs

COMMON_KWARGS = {"var_names": ["K", "C"], "shock_names": ["epsilon"], "param_names": ["alpha", "beta"]}


class TestValidation:
    def test_invalid_initial_condition_raises(self):
        with pytest.raises(ValueError, match="Unknown variables in initial_conditions"):
            validate_perfect_foresight_inputs(
                initial_conditions={"not_a_var": 1.0},
                terminal_conditions={},
                shocks=None,
                param_paths=None,
                simulation_length=10,
                **COMMON_KWARGS,
            )

    def test_invalid_shock_raises(self):
        with pytest.raises(ValueError, match="Unknown shocks"):
            validate_perfect_foresight_inputs(
                initial_conditions={},
                terminal_conditions={},
                shocks={"not_a_shock": np.ones(10)},
                param_paths=None,
                simulation_length=10,
                **COMMON_KWARGS,
            )

    def test_wrong_shock_length_raises(self):
        with pytest.raises(ValueError, match="has length 5, expected 10"):
            validate_perfect_foresight_inputs(
                initial_conditions={},
                terminal_conditions={},
                shocks={"epsilon": np.ones(5)},
                param_paths=None,
                simulation_length=10,
                **COMMON_KWARGS,
            )

    def test_invalid_param_path_raises(self):
        with pytest.raises(ValueError, match="Unknown parameters in param_paths"):
            validate_perfect_foresight_inputs(
                initial_conditions={},
                terminal_conditions={},
                shocks=None,
                param_paths={"not_a_param": 1.0},
                simulation_length=10,
                **COMMON_KWARGS,
            )

    def test_wrong_param_path_length_raises(self):
        with pytest.raises(ValueError, match="param_paths\\['alpha'\\] has length 5, expected 10"):
            validate_perfect_foresight_inputs(
                initial_conditions={},
                terminal_conditions={},
                shocks=None,
                param_paths={"alpha": np.ones(5)},
                simulation_length=10,
                **COMMON_KWARGS,
            )

    def test_scalar_param_path_accepted(self):
        validate_perfect_foresight_inputs(
            initial_conditions={},
            terminal_conditions={},
            shocks=None,
            param_paths={"alpha": 0.5},
            simulation_length=10,
            **COMMON_KWARGS,
        )
