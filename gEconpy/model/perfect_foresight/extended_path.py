import copy
import warnings

from collections.abc import Callable
from functools import partial
from multiprocessing.context import BaseContext
from typing import NamedTuple

import numpy as np
import xarray as xr

from scipy.optimize import OptimizeResult

from gEconpy.model.model import Model
from gEconpy.model.perfect_foresight._parallel import _WORK_COUNT_FIELDS, _map_in_workers, _multiprocessing_context
from gEconpy.model.perfect_foresight.compile import PerfectForesightProblem, compile_perfect_foresight_problem
from gEconpy.model.perfect_foresight.homotopy import _solve_by_homotopy
from gEconpy.model.perfect_foresight.solve import (
    _build_param_matrix,
    _normalize_condition_keys,
    _order_for_problem,
    _settable_parameter_names,
    _solve_stacked,
    _split_parameter_updates,
    _ss_dict_to_array,
)
from gEconpy.model.perfect_foresight.validation import validate_perfect_foresight_inputs
from gEconpy.model.simulate import _draw_shocks
from gEconpy.solvers.sparse_root.base import DEFAULT_TOL, RootSolver
from gEconpy.solvers.sparse_root.direction import ChordDirection
from gEconpy.solvers.sparse_root.line_search import chord

_CHORD_REFRESH_RATIO = 0.1


def simulate_extended_path(
    model: Model,
    n_simulations: int = 1,
    simulation_length: int = 40,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
    random_seed: int | np.random.RandomState | None = None,
    initial_conditions: dict[str, float] | xr.DataArray | None = None,
    horizon: int = 200,
    solver: RootSolver | None = None,
    compile_kwargs: dict | None = None,
    steady_state_kwargs: dict | None = None,
    sparse_root_kwargs: dict | None = None,
    max_homotopy_solves: int = 100,
    cores: int = 1,
    mp_ctx: str | BaseContext | None = None,
) -> tuple[xr.DataArray, xr.Dataset]:
    """
    Simulate trajectories of the nonlinear model by the extended path method.

    Each period draws that period's shock and solves a perfect-foresight problem over ``horizon`` periods, starting
    from the current state, with every later shock set to zero and the steady state as the terminal condition. The
    first period of the solution is kept and becomes the next period's state. Each solve starts from the previous
    solution shifted one period forward, or from the steady state if the model is not finite at that guess. Agents are
    therefore surprised by every shock and expect no further ones, so the trajectories carry the model's nonlinearity
    but no precautionary effects of future uncertainty.

    A period that fails to solve is retried by homotopy. The period's shock and the state's deviation from the steady
    state are scaled by a weight that rises from 0.1 to 1, and each solve starts from the solution at the last weight
    that succeeded.

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
    initial_conditions : dict mapping str to float, or DataArray, optional
        Variable values before the first simulated period, keyed by base name or steady-state name, or a DataArray
        with one dimension over variable names. Variables not listed start at their steady-state values.
    horizon : int, optional
        Number of periods in each period's perfect-foresight problem. It must be long enough for the economy to
        return close to the steady state, which is imposed at its end. Defaults to 200.
    solver : RootSolver, optional
        The root-finding solver for each period's problem. Each trajectory solves with its own copy, so every period of
        a trajectory shares one instance and no state passes between trajectories. Defaults to a
        :func:`~gEconpy.solvers.sparse_root.line_search.chord` solver whose
        :class:`~gEconpy.solvers.sparse_root.direction.ChordDirection` keeps its factorization from one period to the
        next and refreshes it when the residual norm falls by less than a factor of 10 in one step. Pass a solver
        built on another ``ChordDirection`` to change those settings, or
        :func:`~gEconpy.solvers.sparse_root.line_search.newton_armijo` to factorize at every step.
    compile_kwargs : dict, optional
        Keyword arguments forwarded to :func:`pytensor.function` when compiling the model.
    steady_state_kwargs : dict, optional
        Keyword arguments forwarded to ``model.steady_state()``. A parameter value given here also holds in every
        period's problem. ``verbose`` and ``progressbar`` default to False.
    sparse_root_kwargs : dict, optional
        Keyword arguments forwarded to :func:`~gEconpy.solvers.sparse_root.sparse_root.sparse_root` for each
        period's problem. ``f_tol`` defaults to 1e-5 unless ``tol`` is given, and ``progressbar`` defaults to False.
    max_homotopy_solves : int, optional
        Number of solves the homotopy may spend on one failed period. Zero turns homotopy off. Defaults to 100.
    cores : int, optional
        Number of worker processes that simulate trajectories in parallel, each taking a contiguous block of them. The
        output is the same for any number of cores. Defaults to 1, which simulates in the calling process.
    mp_ctx : str or BaseContext, optional
        Start method name, or a context from :func:`multiprocessing.get_context`, for the worker processes when
        ``cores`` is above 1. Defaults to ``"fork"`` on macOS with Apple silicon, ``"forkserver"`` on other macOS,
        and the platform's default elsewhere, except that a JAX compile mode never forks.

    Returns
    -------
    simulations : DataArray
        Simulated levels with dimensions ``(simulation, time, variable)``. A trajectory whose period fails to solve
        even by homotopy is NaN from that period on, with a warning naming the period, and the other trajectories
        continue.
    diagnostics : Dataset
        Per-period solver output with dimensions ``(simulation, time)``: ``success``, the work counts ``nit``,
        ``nfev``, ``njev``, and ``nsolve`` summed over every solve of the period, and ``homotopy_solves``. Periods
        after a failure were not attempted, so ``success`` is False and the counts are zero there.

    Examples
    --------
    .. code-block:: python

        from gEconpy import model_from_gcn, simulate_extended_path
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        paths, diagnostics = simulate_extended_path(model, simulation_length=50, shock_std=0.01, random_seed=0)
        print(bool(diagnostics.success.all()), paths.sel(variable="Y").std(dim="time").values)
    """
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be at least 1, got {n_simulations}")
    if cores < 1:
        raise ValueError(f"cores must be at least 1, got {cores}")

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

    # Every period solves the same problem, so it is compiled and its parameter matrix built once. Inside the loop
    # states follow the problem's variable order, and the output is put back in the model's order at the end.
    problem = compile_perfect_foresight_problem(model, horizon, **(compile_kwargs or {}))
    param_names, deterministic_param_names = _settable_parameter_names(model)
    parameter_updates, _ = _split_parameter_updates(steady_state_kwargs, param_names)
    validate_perfect_foresight_inputs(
        initial_conditions={},
        terminal_conditions={},
        shocks=None,
        param_paths=parameter_updates,
        var_names=problem.var_names,
        shock_names=problem.shock_names,
        param_names=param_names,
        deterministic_param_names=deterministic_param_names,
        simulation_length=horizon,
    )
    params = _build_param_matrix(parameter_updates, problem.param_names, model, horizon)
    shock_columns = [shock_names.index(name) for name in problem.shock_names]

    steady_state = model.steady_state(**steady_state_kwargs)
    steady_state_values = _ss_dict_to_array(steady_state, problem.var_names)
    initial_state = _initial_state(initial_conditions, steady_state_values, list(problem.var_names))

    steady_state_guess = np.tile(steady_state_values, (horizon, 1))
    solver = _order_for_problem(solver if solver is not None else _default_solver(sparse_root_kwargs), problem)
    solve_at_weight = partial(
        _solve_at_weight,
        problem=problem,
        steady_state_values=steady_state_values,
        params=params,
        sparse_root_kwargs=sparse_root_kwargs,
    )

    simulate_trajectory = partial(
        _simulate_trajectory,
        solve_at_weight=solve_at_weight,
        solver=solver,
        initial_state=initial_state,
        steady_state_values=steady_state_values,
        steady_state_guess=steady_state_guess,
        max_homotopy_solves=max_homotopy_solves,
    )
    trajectory_shocks = epsilons[..., shock_columns]
    workers = min(cores, n_simulations)
    if workers == 1:
        trajectories = [simulate_trajectory(shocks=shocks) for shocks in trajectory_shocks]
    else:
        context = _multiprocessing_context(mp_ctx, mode=(compile_kwargs or {}).get("mode"))
        trajectories = _map_in_workers(
            simulate_trajectory, [{"shocks": shocks} for shocks in trajectory_shocks], workers=workers, context=context
        )
    for simulation, trajectory in enumerate(trajectories):
        if trajectory.failure is not None:
            period, message = trajectory.failure
            warnings.warn(
                f"Simulation {simulation} stopped at period {period}, and its values from that period on are NaN. "
                f"{message}",
                stacklevel=2,
            )

    simulations = np.stack([trajectory.states for trajectory in trajectories])
    success = np.stack([trajectory.success for trajectory in trajectories])
    work_counts = {
        field: np.stack([trajectory.work_counts[field] for trajectory in trajectories]) for field in _WORK_COUNT_FIELDS
    }
    homotopy_solves = np.stack([trajectory.homotopy_solves for trajectory in trajectories])

    coords = {"simulation": np.arange(n_simulations), "time": np.arange(simulation_length)}
    diagnostics = xr.Dataset(
        {
            name: (("simulation", "time"), values)
            for name, values in {"success": success, **work_counts, "homotopy_solves": homotopy_solves}.items()
        },
        coords=coords,
    )
    model_order = [problem.var_names.index(name) for name in var_names]
    paths = xr.DataArray(
        simulations[..., model_order], dims=["simulation", "time", "variable"], coords={**coords, "variable": var_names}
    )
    return paths, diagnostics


class _Trajectory(NamedTuple):
    states: np.ndarray
    success: np.ndarray
    work_counts: dict[str, np.ndarray]
    homotopy_solves: np.ndarray
    failure: tuple[int, str] | None


def _simulate_trajectory(
    solve_at_weight: Callable[..., tuple[np.ndarray, OptimizeResult]],
    solver: RootSolver,
    shocks: np.ndarray,
    initial_state: np.ndarray,
    steady_state_values: np.ndarray,
    steady_state_guess: np.ndarray,
    max_homotopy_solves: int,
) -> _Trajectory:
    """
    Simulate one trajectory from ``initial_state`` under ``shocks``, one row per period.

    The trajectory solves with its own copy of ``solver``, so no solver state passes between trajectories.
    """
    simulation_length = shocks.shape[0]
    solve = partial(solve_at_weight, solver=copy.deepcopy(solver))

    states = np.full((simulation_length, initial_state.size), np.nan)
    success = np.zeros(simulation_length, dtype=bool)
    work_counts = {field: np.zeros(simulation_length, dtype=int) for field in _WORK_COUNT_FIELDS}
    homotopy_solves = np.zeros(simulation_length, dtype=int)

    state = initial_state
    guess = steady_state_guess
    for t in range(simulation_length):
        outcome = _solve_by_homotopy(
            solve=partial(solve, state=state, shock=shocks[t]),
            warm_guess=guess,
            base_guess=lambda: steady_state_guess,
            max_homotopy_solves=max_homotopy_solves,
        )

        for field in _WORK_COUNT_FIELDS:
            work_counts[field][t] = outcome.work_counts[field]
        homotopy_solves[t] = outcome.homotopy_solves

        if outcome.path is None:
            return _Trajectory(
                states=states,
                success=success,
                work_counts=work_counts,
                homotopy_solves=homotopy_solves,
                failure=(t, outcome.failure),
            )

        success[t] = True
        state = outcome.path[0]
        guess = np.vstack([outcome.path[1:], steady_state_values])
        states[t] = state

    return _Trajectory(
        states=states, success=success, work_counts=work_counts, homotopy_solves=homotopy_solves, failure=None
    )


def _default_solver(sparse_root_kwargs: dict) -> RootSolver:
    """Return a chord solver that carries its factorization between periods, refreshed when progress slows."""
    x_tol = sparse_root_kwargs.get("x_tol", sparse_root_kwargs.get("tol", DEFAULT_TOL))
    direction = ChordDirection(
        recompute_every=None,
        refresh_ratio=_CHORD_REFRESH_RATIO,
        refresh_step_tol=x_tol,
        keep_across_solves=True,
    )
    return chord(direction=direction)


def _solve_at_weight(
    weight: float,
    x0: np.ndarray,
    state: np.ndarray,
    shock: np.ndarray,
    problem: PerfectForesightProblem,
    steady_state_values: np.ndarray,
    params: np.ndarray,
    solver: RootSolver | None,
    sparse_root_kwargs: dict,
) -> tuple[np.ndarray, OptimizeResult]:
    """Solve the period's problem with its shock and initial deviation from the steady state scaled by ``weight``."""
    shocks = np.zeros((problem.T, problem.n_shocks))
    shocks[0] = weight * shock

    result = _solve_stacked(
        problem=problem,
        x0=x0.ravel(),
        y_initial=steady_state_values + weight * (state - steady_state_values),
        y_terminal=steady_state_values,
        shocks=shocks,
        params=params,
        solver=solver,
        sparse_root_kwargs=sparse_root_kwargs,
    )
    return result.x.reshape(problem.T, problem.n_vars), result


def _initial_state(
    initial_conditions: dict[str, float] | xr.DataArray | None,
    steady_state_values: np.ndarray,
    var_names: list[str],
) -> np.ndarray:
    """Fill the steady state with the user's initial conditions, rejecting names that are not model variables."""
    conditions = _normalize_condition_keys(initial_conditions)
    unknown = sorted(set(conditions) - set(var_names))
    if unknown:
        raise ValueError(f"initial_conditions has unknown variables: {', '.join(unknown)}")

    state = steady_state_values.copy()
    for name, value in conditions.items():
        state[var_names.index(name)] = value
    return state
