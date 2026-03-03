import numpy as np


def validate_perfect_foresight_inputs(
    initial_conditions: dict[str, float],
    terminal_conditions: dict[str, float],
    shocks: dict[str, np.ndarray] | None,
    param_paths: dict[str, float | np.ndarray] | None,
    var_names: list[str],
    shock_names: list[str],
    param_names: list[str],
    simulation_length: int,
) -> None:
    """Validate inputs to solve_perfect_foresight.

    Parameters
    ----------
    initial_conditions : dict of str to float
        Initial values for state variables at t=-1.
    terminal_conditions : dict of str to float
        Terminal values for state variables at t=T.
    shocks : dict of str to ndarray or None
        Shock paths over the simulation horizon.
    param_paths : dict of str to float or ndarray, or None
        Parameter overrides (scalar or time-varying).
    var_names : list of str
        Valid variable names from the model.
    shock_names : list of str
        Valid shock names from the model.
    param_names : list of str
        Valid parameter names from the model.
    simulation_length : int
        Number of time periods.
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
            if isinstance(value, np.ndarray) and len(value) != simulation_length:
                raise ValueError(f"param_paths['{name}'] has length {len(value)}, expected {simulation_length}")
