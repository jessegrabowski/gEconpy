from collections.abc import Sequence

import numpy as np


def validate_perfect_foresight_inputs(
    initial_conditions: dict[str, float],
    terminal_conditions: dict[str, float],
    shocks: dict[str, np.ndarray] | None,
    param_paths: dict[str, float | Sequence[float] | np.ndarray] | None,
    var_names: list[str],
    shock_names: list[str],
    param_names: list[str],
    simulation_length: int,
) -> None:
    """
    Check that every user-supplied name is known to the model and every path has the simulation length.

    Parameters
    ----------
    initial_conditions : dict mapping str to float
        Variable values at ``t = -1``, keyed by base name.
    terminal_conditions : dict mapping str to float
        Variable values at ``t = T``, keyed by base name.
    shocks : dict mapping str to ndarray, optional
        Shock paths over the horizon, keyed by shock name.
    param_paths : dict mapping str to float, sequence of float, or ndarray, optional
        Parameter overrides, keyed by parameter name. A scalar holds for every period. A list or array gives one
        value per period.
    var_names : list of str
        Variable names the model knows.
    shock_names : list of str
        Shock names the model knows.
    param_names : list of str
        Parameter names the model knows.
    simulation_length : int
        Number of periods.
    """
    var_set = set(var_names)
    shock_set = set(shock_names)
    param_set = set(param_names)

    invalid_initial = set(initial_conditions.keys()) - var_set
    if invalid_initial:
        raise ValueError(f"Unknown variables in initial_conditions: {invalid_initial}. Valid: {var_names}")

    invalid_terminal = set(terminal_conditions.keys()) - var_set
    if invalid_terminal:
        raise ValueError(f"Unknown variables in terminal_conditions: {invalid_terminal}. Valid: {var_names}")

    if shocks:
        invalid_shocks = set(shocks.keys()) - shock_set
        if invalid_shocks:
            raise ValueError(f"Unknown shocks: {invalid_shocks}. Valid: {shock_names}")

        for name, values in shocks.items():
            if len(values) != simulation_length:
                raise ValueError(f"Shock '{name}' has length {len(values)}, expected {simulation_length}")

    if param_paths:
        invalid_params = set(param_paths.keys()) - param_set
        if invalid_params:
            raise ValueError(f"Unknown parameters in param_paths: {invalid_params}. Valid: {param_names}")

        for name, value in param_paths.items():
            if np.ndim(value) > 0 and len(value) != simulation_length:
                raise ValueError(f"param_paths['{name}'] has length {len(value)}, expected {simulation_length}")
