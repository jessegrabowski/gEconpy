import numpy as np
import pytest

from gEconpy.model.perfect_foresight.validation import validate_perfect_foresight_inputs


class TestValidation:
    def test_invalid_initial_condition_raises(self):
        with pytest.raises(ValueError, match="Unknown variables in initial_conditions"):
            validate_perfect_foresight_inputs(
                initial_conditions={"not_a_var": 1.0},
                terminal_conditions={},
                shocks=None,
                var_names=["K", "C"],
                shock_names=["epsilon"],
                T=10,
            )

    def test_invalid_shock_raises(self):
        with pytest.raises(ValueError, match="Unknown shocks"):
            validate_perfect_foresight_inputs(
                initial_conditions={},
                terminal_conditions={},
                shocks={"not_a_shock": np.ones(10)},
                var_names=["K", "C"],
                shock_names=["epsilon"],
                T=10,
            )

    def test_wrong_shock_length_raises(self):
        with pytest.raises(ValueError, match="has length 5, expected 10"):
            validate_perfect_foresight_inputs(
                initial_conditions={},
                terminal_conditions={},
                shocks={"epsilon": np.ones(5)},
                var_names=["K", "C"],
                shock_names=["epsilon"],
                T=10,
            )
