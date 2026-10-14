import numpy as np
import xarray as xr

from gEconpy.model.model import Model
from gEconpy.model.perfect_foresight.solve import (
    _normalize_condition_keys,
    _ss_dict_to_array,
    solve_perfect_foresight,
)
from gEconpy.model.simulate import _draw_shocks
from gEconpy.solvers.sparse_root.base import RootSolver

_WORK_COUNT_FIELDS = ("nit", "nfev", "njev", "nsolve")


def simulate_extended_path(
    model: Model,
    n_simulations: int = 1,
    simulation_length: int = 40,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
    random_seed: int | np.random.RandomState | None = None,
    initial_conditions: dict[str, float] | None = None,
    horizon: int = 200,
    solver: RootSolver | None = None,
    compile_kwargs: dict | None = None,
    steady_state_kwargs: dict | None = None,
    sparse_root_kwargs: dict | None = None,
) -> tuple[xr.DataArray, xr.Dataset]:
    """
    Simulate trajectories of the nonlinear model by the extended path method.

    Each period draws that period's shock and solves a perfect-foresight problem over ``horizon`` periods, starting
    from the current state, with every later shock set to zero and the steady state as the terminal condition. The
    first period of the solution is kept and becomes the next period's state. Agents are therefore surprised by
    every shock and expect no further ones, so the trajectories carry the model's nonlinearity but no precautionary
    effects of future uncertainty.

    Shocks are drawn exactly as :func:`~gEconpy.model.simulate.simulate` draws them, so the two simulators see the
    same shocks for the same arguments and seed.

    Parameters
    ----------
    model : Model
        The DSGE model to simulate.
    n_simulations : int, optional
        Number of trajectories to simulate. Defaults to 1.
    simulation_length : int, optional
        Length of each simulated trajectory. Defaults to 40.
    shock_std_dict : dict mapping str to float, optional
        Standard deviation of each shock, keyed by shock name. Defaults to None.
    shock_cov_matrix : ndarray, optional
        Covariance matrix of the shocks, of shape ``(n_shocks, n_shocks)``. Defaults to None.
    shock_std : float or sequence of float, optional
        Standard deviation shared by every shock, or one standard deviation per shock in model order. Defaults to
        None.
    random_seed : int, RandomState, or Generator, optional
        Seed for the shock draws. Defaults to None.
    initial_conditions : dict mapping str to float, optional
        Variable values before the first simulated period, keyed by base name or steady-state name. Variables not
        listed start at their steady-state values.
    horizon : int, optional
        Number of periods in each period's perfect-foresight problem. It must be long enough for the economy to
        return close to the steady state, which is imposed at its end. Defaults to 200.
    solver : RootSolver, optional
        The root-finding solver for each period's problem. Defaults to the default of
        :func:`~gEconpy.model.perfect_foresight.solve.solve_perfect_foresight`.
    compile_kwargs : dict, optional
        Keyword arguments forwarded to :func:`pytensor.function` when compiling the model.
    steady_state_kwargs : dict, optional
        Keyword arguments forwarded to ``model.steady_state()``. ``verbose`` and ``progressbar`` default to False.
    sparse_root_kwargs : dict, optional
        Keyword arguments forwarded to :func:`~gEconpy.solvers.sparse_root.sparse_root.sparse_root` for each
        period's problem. ``f_tol`` defaults to 1e-5 unless ``tol`` is given, and ``progressbar`` defaults to False.

    Returns
    -------
    simulations : DataArray
        Simulated levels with dimensions ``(simulation, time, variable)``.
    diagnostics : Dataset
        Per-period solver output with dimensions ``(simulation, time)``: ``success``, and the work counts ``nit``,
        ``nfev``, ``njev``, and ``nsolve``.

    Examples
    --------
    .. code-block:: python

        from gEconpy import model_from_gcn, simulate_extended_path
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        paths, diagnostics = simulate_extended_path(model, simulation_length=50, shock_std=0.01, random_seed=0)
        print(bool(diagnostics.success.all()), paths.sel(variable="Y").std(dim="time").values)
    """
    steady_state_kwargs = {"verbose": False, "progressbar": False, **(steady_state_kwargs or {})}
    sparse_root_kwargs = {"progressbar": False, **(sparse_root_kwargs or {})}
    if "tol" not in sparse_root_kwargs:
        sparse_root_kwargs.setdefault("f_tol", 1e-5)

    var_names = [variable.base_name for variable in model.variables]
    shock_names = [shock.base_name for shock in model.shocks]

    epsilons = _draw_shocks(
        model=model,
        n_simulations=n_simulations,
        simulation_length=simulation_length,
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        random_seed=random_seed,
    )

    steady_state = model.steady_state(**steady_state_kwargs)
    steady_state_values = _ss_dict_to_array(steady_state, var_names)
    terminal_conditions = dict(zip(var_names, steady_state_values, strict=True))
    initial_state = _initial_state(initial_conditions, steady_state_values, var_names)

    simulations = np.full((n_simulations, simulation_length, len(var_names)), np.nan)
    success = np.zeros((n_simulations, simulation_length), dtype=bool)
    work_counts = {field: np.zeros((n_simulations, simulation_length), dtype=int) for field in _WORK_COUNT_FIELDS}

    for simulation in range(n_simulations):
        state = initial_state
        guess = np.tile(steady_state_values, (horizon, 1))

        for t in range(simulation_length):
            shock_paths = np.zeros((horizon, len(shock_names)))
            shock_paths[0] = epsilons[simulation, t]

            solution, result = solve_perfect_foresight(
                model,
                simulation_length=horizon,
                x0=guess,
                initial_conditions=dict(zip(var_names, state, strict=True)),
                terminal_conditions=terminal_conditions,
                shocks=dict(zip(shock_names, shock_paths.T, strict=True)),
                compile_kwargs=compile_kwargs,
                solver=solver,
                sparse_root_kwargs=sparse_root_kwargs,
            )

            success[simulation, t] = result.success
            for field in _WORK_COUNT_FIELDS:
                work_counts[field][simulation, t] = result[field]

            path = solution[var_names].to_numpy()
            state = path[0]
            guess = np.vstack([path[1:], steady_state_values])
            simulations[simulation, t] = state

    coords = {"simulation": np.arange(n_simulations), "time": np.arange(simulation_length)}
    diagnostics = xr.Dataset(
        {name: (("simulation", "time"), values) for name, values in {"success": success, **work_counts}.items()},
        coords=coords,
    )
    paths = xr.DataArray(simulations, dims=["simulation", "time", "variable"], coords={**coords, "variable": var_names})
    return paths, diagnostics


def _initial_state(
    initial_conditions: dict[str, float] | None,
    steady_state_values: np.ndarray,
    var_names: list[str],
) -> np.ndarray:
    """Fill the steady state with the user's initial conditions, rejecting names that are not model variables."""
    conditions = _normalize_condition_keys(initial_conditions or {})
    unknown = sorted(set(conditions) - set(var_names))
    if unknown:
        raise ValueError(f"initial_conditions has unknown variables: {', '.join(unknown)}")

    state = steady_state_values.copy()
    for name, value in conditions.items():
        state[var_names.index(name)] = value
    return state
