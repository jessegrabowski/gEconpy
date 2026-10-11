import numpy as np
import pytest

from gEconpy.model.perfect_foresight.validation import validate_perfect_foresight_inputs

MODEL_NAMES = {
    "var_names": ["K", "C"],
    "shock_names": ["epsilon"],
    "param_names": ["alpha", "beta", "rstar"],
    "deterministic_param_names": ["rstar"],
}
NO_INPUTS = {"initial_conditions": {}, "terminal_conditions": {}, "shocks": None, "param_paths": None}


@pytest.mark.parametrize(
    "overrides, simulation_length, match",
    [
        ({"initial_conditions": {"not_a_var": 1.0}}, 10, "Unknown variables in initial_conditions"),
        ({"terminal_conditions": {"not_a_var": 1.0}}, 10, "Unknown variables in terminal_conditions"),
        ({"shocks": {"not_a_shock": np.ones(10)}}, 10, "Unknown shocks"),
        ({"shocks": {"epsilon": np.ones(5)}}, 10, "has length 5, expected 10"),
        ({"shocks": {"epsilon": 0.1}}, 10, "must be a one-dimensional path"),
        ({"shocks": {"epsilon": np.ones((10, 1))}}, 10, "must be a one-dimensional path"),
        ({"param_paths": {"not_a_param": 1.0}}, 10, "Unknown parameters in param_paths"),
        ({"param_paths": {"alpha": np.ones(5)}}, 10, r"param_paths\['alpha'\] has length 5, expected 10"),
        ({"param_paths": {"alpha": [0.5] * 5}}, 10, r"param_paths\['alpha'\] has length 5, expected 10"),
        ({"param_paths": {"alpha": np.ones((10, 1))}}, 10, r"param_paths\['alpha'\] must be a scalar"),
        ({"param_paths": {"rstar": 0.05}}, 10, "Cannot set a path for deterministic parameters: rstar"),
        ({}, 0, "simulation_length must be at least 1"),
    ],
    ids=[
        "unknown-initial-variable",
        "unknown-terminal-variable",
        "unknown-shock",
        "wrong-shock-length",
        "scalar-shock",
        "two-dimensional-shock",
        "unknown-parameter",
        "wrong-parameter-path-length",
        "wrong-parameter-list-length",
        "two-dimensional-parameter-path",
        "deterministic-parameter-path",
        "non-positive-simulation-length",
    ],
)
def test_invalid_inputs_raise(overrides, simulation_length, match):
    with pytest.raises(ValueError, match=match):
        validate_perfect_foresight_inputs(
            **{**NO_INPUTS, **overrides}, simulation_length=simulation_length, **MODEL_NAMES
        )


def test_unknown_parameter_message_lists_only_the_free_parameters():
    with pytest.raises(ValueError, match=r"Valid: \['alpha', 'beta'\]"):
        validate_perfect_foresight_inputs(
            **{**NO_INPUTS, "param_paths": {"not_a_param": 1.0}}, simulation_length=10, **MODEL_NAMES
        )


def test_scalar_param_path_accepted():
    validate_perfect_foresight_inputs(
        **{**NO_INPUTS, "param_paths": {"alpha": 0.5}}, simulation_length=10, **MODEL_NAMES
    )
