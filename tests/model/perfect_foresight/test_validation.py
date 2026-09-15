import numpy as np
import pytest

from gEconpy.model.perfect_foresight.validation import validate_perfect_foresight_inputs

MODEL_NAMES = {"var_names": ["K", "C"], "shock_names": ["epsilon"], "param_names": ["alpha", "beta"]}
NO_INPUTS = {"initial_conditions": {}, "terminal_conditions": {}, "shocks": None, "param_paths": None}


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"initial_conditions": {"not_a_var": 1.0}}, "Unknown variables in initial_conditions"),
        ({"terminal_conditions": {"not_a_var": 1.0}}, "Unknown variables in terminal_conditions"),
        ({"shocks": {"not_a_shock": np.ones(10)}}, "Unknown shocks"),
        ({"shocks": {"epsilon": np.ones(5)}}, "has length 5, expected 10"),
        ({"param_paths": {"not_a_param": 1.0}}, "Unknown parameters in param_paths"),
        ({"param_paths": {"alpha": np.ones(5)}}, r"param_paths\['alpha'\] has length 5, expected 10"),
        ({"param_paths": {"alpha": [0.5] * 5}}, r"param_paths\['alpha'\] has length 5, expected 10"),
    ],
    ids=[
        "unknown-initial-variable",
        "unknown-terminal-variable",
        "unknown-shock",
        "wrong-shock-length",
        "unknown-parameter",
        "wrong-parameter-path-length",
        "wrong-parameter-list-length",
    ],
)
def test_invalid_inputs_raise(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_perfect_foresight_inputs(**{**NO_INPUTS, **overrides}, simulation_length=10, **MODEL_NAMES)


def test_scalar_param_path_accepted():
    validate_perfect_foresight_inputs(
        **{**NO_INPUTS, "param_paths": {"alpha": 0.5}}, simulation_length=10, **MODEL_NAMES
    )
