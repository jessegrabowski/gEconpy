from collections.abc import Sequence

import numpy as np


def validate_perfect_foresight_inputs(
    *,
    initial_conditions: dict[str, float],
    terminal_conditions: dict[str, float],
    shocks: dict[str, np.ndarray] | None,
    param_paths: dict[str, float | Sequence[float] | np.ndarray] | None,
    var_names: Sequence[str],
    shock_names: Sequence[str],
    param_names: Sequence[str],
    deterministic_param_names: Sequence[str],
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
    var_names : sequence of str
        Variable names the model knows.
    shock_names : sequence of str
        Shock names the model knows.
    param_names : sequence of str
        Parameter names the model knows.
    deterministic_param_names : sequence of str
        Names in ``param_names`` that the model derives from other parameters, which cannot be set directly.
    simulation_length : int
        Number of periods.

    Raises
    ------
    ValueError
        If a name is unknown, if a path does not have the simulation length, if a deterministic parameter is
        given a path, or if ``simulation_length`` is not positive.
    """
    if simulation_length < 1:
        raise ValueError(f"simulation_length must be at least 1, got {simulation_length}")

    var_set = set(var_names)
    shock_set = set(shock_names)

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
            if np.ndim(values) != 1:
                raise ValueError(
                    f"Shock '{name}' must be a one-dimensional path of length {simulation_length}, got an input "
                    f"with {np.ndim(values)} dimensions"
                )
            if len(values) != simulation_length:
                raise ValueError(f"Shock '{name}' has length {len(values)}, expected {simulation_length}")

    if param_paths:
        deterministic_set = set(deterministic_param_names)
        free_param_names = [name for name in param_names if name not in deterministic_set]

        invalid_params = set(param_paths.keys()) - set(param_names)
        if invalid_params:
            raise ValueError(f"Unknown parameters in param_paths: {invalid_params}. Valid: {free_param_names}")

        # A deterministic parameter is a function of the free parameters, so the model recomputes it from them at
        # every update. Writing a path for it directly would be honored in the simulated periods and ignored when
        # the boundary steady states are computed, leaving the two halves of the problem disagreeing.
        invalid_deterministic = sorted(set(param_paths.keys()) & deterministic_set)
        if invalid_deterministic:
            raise ValueError(
                f"Cannot set a path for deterministic parameters: {', '.join(invalid_deterministic)}. They are "
                f"derived from other parameters, so set a path for those instead. Free parameters: "
                f"{free_param_names}"
            )

        for name, value in param_paths.items():
            if np.ndim(value) > 1:
                raise ValueError(
                    f"param_paths['{name}'] must be a scalar or a one-dimensional path, got an input with "
                    f"{np.ndim(value)} dimensions"
                )
            if np.ndim(value) == 1 and len(value) != simulation_length:
                raise ValueError(f"param_paths['{name}'] has length {len(value)}, expected {simulation_length}")
