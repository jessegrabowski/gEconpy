import warnings

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import numpy as np
import xarray as xr

from scipy.optimize import OptimizeResult

from gEconpy.exceptions import NonFiniteInitialGuessError
from gEconpy.model.model import Model
from gEconpy.model.perfect_foresight.compile import PerfectForesightProblem, compile_perfect_foresight_problem
from gEconpy.model.perfect_foresight.solve import (
    _build_param_matrix,
    _normalize_condition_keys,
    _solve_stacked,
    _ss_dict_to_array,
)
from gEconpy.model.perfect_foresight.validation import validate_perfect_foresight_inputs
from gEconpy.model.simulate import _draw_shocks
from gEconpy.solvers.sparse_root.base import RootSolver

_WORK_COUNT_FIELDS = ("nit", "nfev", "njev", "nsolve")
_HOMOTOPY_INITIAL_STEP = 0.1
_HOMOTOPY_MIN_STEP = 1e-4


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
    max_homotopy_solves: int = 100,
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
        Keyword arguments forwarded to ``model.steady_state()``. A parameter value given here also holds in every
        period's problem. ``verbose`` and ``progressbar`` default to False.
    sparse_root_kwargs : dict, optional
        Keyword arguments forwarded to :func:`~gEconpy.solvers.sparse_root.sparse_root.sparse_root` for each
        period's problem. ``f_tol`` defaults to 1e-5 unless ``tol`` is given, and ``progressbar`` defaults to False.
    max_homotopy_solves : int, optional
        Number of solves the homotopy may spend on one failed period. Zero turns homotopy off. Defaults to 100.

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
    parameter_updates = {name: value for name, value in steady_state_kwargs.items() if name in problem.param_names}
    validate_perfect_foresight_inputs(
        initial_conditions={},
        terminal_conditions={},
        shocks=None,
        param_paths=parameter_updates,
        var_names=problem.var_names,
        shock_names=problem.shock_names,
        param_names=problem.param_names,
        deterministic_param_names=problem.deterministic_param_names,
        simulation_length=horizon,
    )
    params = _build_param_matrix(parameter_updates, problem.param_names, model, horizon)
    shock_columns = [shock_names.index(name) for name in problem.shock_names]

    steady_state = model.steady_state(**steady_state_kwargs)
    steady_state_values = _ss_dict_to_array(steady_state, problem.var_names)
    initial_state = _initial_state(initial_conditions, steady_state_values, list(problem.var_names))

    simulations = np.full((n_simulations, simulation_length, len(var_names)), np.nan)
    success = np.zeros((n_simulations, simulation_length), dtype=bool)
    work_counts = {field: np.zeros((n_simulations, simulation_length), dtype=int) for field in _WORK_COUNT_FIELDS}
    homotopy_solves = np.zeros((n_simulations, simulation_length), dtype=int)

    steady_state_guess = np.tile(steady_state_values, (horizon, 1))
    solve_at_weight = partial(
        _solve_at_weight,
        problem=problem,
        steady_state_values=steady_state_values,
        params=params,
        solver=solver,
        sparse_root_kwargs=sparse_root_kwargs,
    )

    for simulation in range(n_simulations):
        state = initial_state
        guess = steady_state_guess

        for t in range(simulation_length):
            outcome = _solve_period(
                solve=partial(solve_at_weight, state=state, shock=epsilons[simulation, t, shock_columns]),
                warm_guess=guess,
                steady_state_guess=steady_state_guess,
                max_homotopy_solves=max_homotopy_solves,
            )

            for field in _WORK_COUNT_FIELDS:
                work_counts[field][simulation, t] = outcome.work_counts[field]
            homotopy_solves[simulation, t] = outcome.homotopy_solves

            if outcome.path is None:
                warnings.warn(
                    f"Simulation {simulation} stopped at period {t}, and its values from that period on are NaN. "
                    f"{outcome.failure}",
                    stacklevel=2,
                )
                break

            success[simulation, t] = True
            state = outcome.path[0]
            guess = np.vstack([outcome.path[1:], steady_state_values])
            simulations[simulation, t] = state

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


class _PeriodOutcome(NamedTuple):
    path: np.ndarray | None
    work_counts: dict[str, int]
    homotopy_solves: int
    failure: str | None


def _solve_period(
    solve: Callable[..., tuple[np.ndarray, OptimizeResult]],
    warm_guess: np.ndarray,
    steady_state_guess: np.ndarray,
    max_homotopy_solves: int,
) -> _PeriodOutcome:
    """
    Solve one period from the warm start, and by homotopy if that fails.

    The homotopy scales the period's shock and the initial state's deviation from the steady state by a weight. At
    weight 0 the steady state solves the problem, so the weight rises from there to 1, each solve starting from the
    last solution. The step doubles after a success and halves after a failure.
    """
    work_counts: dict[str, int] = dict.fromkeys(_WORK_COUNT_FIELDS, 0)

    def attempt(weight: float, guess: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        try:
            solution, result = _solve_from(solve, weight=weight, guess=guess, steady_state_guess=steady_state_guess)
        except NonFiniteInitialGuessError as error:
            return None, str(error)

        for field in _WORK_COUNT_FIELDS:
            work_counts[field] += result[field]
        if not result.success:
            return None, f"The solver reported: {result.message}"
        return solution, None

    path, failure = attempt(weight=1.0, guess=warm_guess)
    if path is not None:
        return _PeriodOutcome(path=path, work_counts=work_counts, homotopy_solves=0, failure=None)

    accepted_weight = 0.0
    step = _HOMOTOPY_INITIAL_STEP
    guess = steady_state_guess
    homotopy_solves = 0
    while homotopy_solves < max_homotopy_solves and step >= _HOMOTOPY_MIN_STEP:
        weight = min(accepted_weight + step, 1.0)
        homotopy_solves += 1
        path, _ = attempt(weight=weight, guess=guess)

        if path is None:
            step /= 2
        elif weight == 1.0:
            return _PeriodOutcome(path=path, work_counts=work_counts, homotopy_solves=homotopy_solves, failure=None)
        else:
            accepted_weight, guess = weight, path
            step *= 2

    failure = f"{failure.rstrip('.')}. Homotopy reached weight {accepted_weight:.3g} in {homotopy_solves} solves."
    return _PeriodOutcome(path=None, work_counts=work_counts, homotopy_solves=homotopy_solves, failure=failure)


def _solve_from(
    solve: Callable[..., tuple[np.ndarray, OptimizeResult]],
    weight: float,
    guess: np.ndarray,
    steady_state_guess: np.ndarray,
) -> tuple[np.ndarray, OptimizeResult]:
    try:
        return solve(weight=weight, x0=guess)
    except NonFiniteInitialGuessError:
        if guess is steady_state_guess:
            raise
        return solve(weight=weight, x0=steady_state_guess)


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
