import copy
import warnings

from collections.abc import Sequence
from functools import partial
from multiprocessing.context import BaseContext
from typing import NamedTuple, TypedDict, TypeGuard

import numpy as np
import xarray as xr

from gEconpy.exceptions import NonFiniteInitialGuessError
from gEconpy.model.model import Model
from gEconpy.model.perfect_foresight._parallel import _WORK_COUNT_FIELDS, _map_in_workers, _multiprocessing_context
from gEconpy.model.perfect_foresight.compile import PerfectForesightProblem, compile_perfect_foresight_problem
from gEconpy.model.perfect_foresight.solve import (
    _boundary_vector,
    _build_param_matrix,
    _build_shock_matrix,
    _check_all_finite,
    _normalize_condition_keys,
    _order_for_problem,
    _settable_parameter_names,
    _solve_stacked,
    _split_parameter_updates,
    _ss_dict_to_array,
)
from gEconpy.model.perfect_foresight.validation import validate_perfect_foresight_inputs
from gEconpy.model.tables import SAMPLE_DIMS, group_dataset
from gEconpy.solvers.sparse_root import newton_armijo
from gEconpy.solvers.sparse_root.base import RootSolver


class _DrawInput(TypedDict):
    parameters: dict[str, float]
    conditions: dict[str, float]


class _DrawOutcome(NamedTuple):
    path: np.ndarray | None
    steady_state_solved: bool
    work_counts: dict[str, int]
    failure: str | None


def sample_perfect_foresight(
    model: Model,
    idata: xr.DataTree | xr.Dataset,
    simulation_length: int,
    group: str = "posterior",
    shocks: dict[str, np.ndarray] | None = None,
    initial_conditions: dict[str, float] | xr.DataArray | None = None,
    solver: RootSolver | None = None,
    compile_kwargs: dict | None = None,
    steady_state_kwargs: dict | None = None,
    sparse_root_kwargs: dict | None = None,
    cores: int = 1,
    mp_ctx: str | BaseContext | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Solve the nonlinear model under perfect foresight at every parameter draw of a trace.

    Each draw solves the problem :func:`~gEconpy.model.perfect_foresight.solve.solve_perfect_foresight` solves, at
    that draw's parameters, from that draw's own steady state to the same steady state at the end of the horizon.
    The perfect-foresight problem is compiled once and shared by every draw, while each draw solves its own steady
    state. Each draw's Newton solve starts from that steady state, so a draw's path does not depend on which other
    draws are solved or in what order.

    Parameters
    ----------
    model : Model
        The DSGE model to solve.
    idata : DataTree or Dataset
        Sampler output whose ``group`` holds draws of the model's free parameters, or that group itself. A free
        parameter without draws keeps its model value.
    simulation_length : int
        Number of periods in each draw's path.
    group : str, optional
        Group of ``idata`` holding the parameter draws. Defaults to ``"posterior"``.
    shocks : dict mapping str to ndarray, optional
        Shock paths over the horizon, keyed by shock name, each of length ``simulation_length``, shared by every
        draw. Shocks not listed are zero. Defaults to None.
    initial_conditions : dict mapping str to float, or DataArray, optional
        Variable values at ``t = -1``, keyed by base name or steady-state name. A dict, or a DataArray over variable
        names alone, is shared by every draw. A DataArray with ``chain`` and ``draw`` dimensions gives each draw its
        own values, such as :meth:`~gEconpy.model.statespace.DSGEStateSpace.deviations_to_levels` output at one date.
        Variables not listed start at the draw's steady state. Defaults to None.
    solver : RootSolver, optional
        The root-finding solver for each draw's problem. Each draw solves with its own copy. Defaults to
        :func:`~gEconpy.solvers.sparse_root.line_search.newton_armijo`.
    compile_kwargs : dict, optional
        Keyword arguments forwarded to :func:`pytensor.function` when compiling the model. Defaults to None.
    steady_state_kwargs : dict, optional
        Keyword arguments forwarded to ``model.steady_state()`` at every draw. A parameter value given here holds at
        every draw unless the draws carry the same parameter. ``verbose`` and ``progressbar`` default to False.
    sparse_root_kwargs : dict, optional
        Keyword arguments forwarded to :func:`~gEconpy.solvers.sparse_root.sparse_root.sparse_root` for each draw.
        ``progressbar`` defaults to False.
    cores : int, optional
        Number of worker processes that solve draws in parallel, each taking a contiguous block of them. The output is
        the same for any number of cores.
        Defaults to 1, which solves in the calling process.
    mp_ctx : str or BaseContext, optional
        Start method name, or a context from :func:`multiprocessing.get_context`, for the worker processes when
        ``cores`` is above 1. Defaults as for
        :func:`~gEconpy.model.perfect_foresight.extended_path.simulate_extended_path`.

    Returns
    -------
    paths : Dataset
        One variable per model variable, in levels, with dimensions ``(chain, draw, time)``. A draw that fails to
        solve is NaN, and a warning reports how many did. xarray reductions such as ``mean`` skip NaN by default, so
        a summary over ``paths`` covers only the draws that solved unless ``skipna=False`` is passed. The paths of
        every draw are held in memory at once, and the worker results are copied once more while they are combined.
    diagnostics : Dataset
        Per-draw solver output with dimensions ``(chain, draw)``: ``success``, ``steady_state_success``, which is
        False where the draw failed before Newton started, and the work counts ``nit``, ``nfev``, ``njev`` and
        ``nsolve``.

    Examples
    --------
    Trace a shock to capital across the posterior, starting each draw from its smoothed state in 2014Q4:

    .. code-block:: python

        smoothed = ss_mod.sample_conditional_posterior(idata)["smoothed_posterior"]
        start = ss_mod.deviations_to_levels(smoothed, idata).sel(time="2014-10-01")
        paths, diagnostics = ge.sample_perfect_foresight(
            model, idata, simulation_length=400, shocks={"epsilon_xi_K": shock_path}, initial_conditions=start
        )
        paths["C"].mean(("chain", "draw"))
    """
    if cores < 1:
        raise ValueError(f"cores must be at least 1, got {cores}")

    draws = group_dataset(idata, group=group)
    free_names = [parameter.name for parameter in model.params]
    drawn_names = [name for name in free_names if name in draws.data_vars]
    if not drawn_names:
        raise ValueError(
            "The parameter draws hold none of the model's free parameters, so every draw would solve the same "
            f"problem. The free parameters are: {', '.join(free_names)}."
        )
    steady_state_kwargs = {"verbose": False, "progressbar": False, **(steady_state_kwargs or {})}
    param_names, deterministic_param_names = _settable_parameter_names(model)
    fixed_parameters, steady_state_options = _split_parameter_updates(steady_state_kwargs, param_names)
    sparse_root_kwargs = {"progressbar": False, **(sparse_root_kwargs or {})}

    var_names = [variable.base_name for variable in model.variables]
    validate_perfect_foresight_inputs(
        initial_conditions={},
        terminal_conditions={},
        shocks=shocks,
        param_paths=fixed_parameters,
        var_names=var_names,
        shock_names=[shock.base_name for shock in model.shocks],
        param_names=param_names,
        deterministic_param_names=deterministic_param_names,
        simulation_length=simulation_length,
    )
    _check_start_covers_draws(initial_conditions, draws=draws)
    _check_all_finite(
        shocks=np.asarray(list((shocks or {}).values()), dtype=float),
        steady_state_kwargs=np.asarray(list(fixed_parameters.values()), dtype=float),
    )

    drawn_values = {name: draws[name].transpose(*SAMPLE_DIMS).values for name in drawn_names}
    predetermined = set(model.predetermined_variables)
    draw_inputs: list[_DrawInput] = [
        {
            "parameters": fixed_parameters
            | {name: float(values[chain_index, draw_index]) for name, values in drawn_values.items()},
            "conditions": _draw_conditions(
                initial_conditions, chain=chain, draw=draw, var_names=var_names, predetermined=predetermined
            ),
        }
        for chain_index, chain in enumerate(draws["chain"].values)
        for draw_index, draw in enumerate(draws["draw"].values)
    ]

    problem = compile_perfect_foresight_problem(model, simulation_length, **(compile_kwargs or {}))
    shock_matrix = _build_shock_matrix(shocks=shocks, shock_names=problem.shock_names, T=simulation_length)
    solve_draw = partial(
        _solve_draw,
        model=model,
        problem=problem,
        shock_matrix=shock_matrix,
        steady_state_options=steady_state_options,
        solver=_order_for_problem(solver if solver is not None else newton_armijo(), problem),
        sparse_root_kwargs=sparse_root_kwargs,
    )

    workers = min(cores, len(draw_inputs))
    if workers == 1:
        outcomes = [solve_draw(**inputs) for inputs in draw_inputs]
    else:
        context = _multiprocessing_context(mp_ctx, mode=(compile_kwargs or {}).get("mode"))
        outcomes = _map_in_workers(solve_draw, draw_inputs, workers=workers, context=context)

    return _assemble(outcomes, draws=draws, problem=problem, model=model)


def _has_draws(initial_conditions: dict[str, float] | xr.DataArray | None) -> TypeGuard[xr.DataArray]:
    return isinstance(initial_conditions, xr.DataArray) and set(SAMPLE_DIMS) <= set(initial_conditions.dims)


def _check_start_covers_draws(initial_conditions: dict[str, float] | xr.DataArray | None, draws: xr.Dataset) -> None:
    """Reject a per-draw start that lacks a chain or draw of the trace, which ``sel`` would report as a KeyError."""
    if not _has_draws(initial_conditions):
        return
    missing = {
        dim: sorted(set(draws[dim].values.tolist()) - set(initial_conditions[dim].values.tolist()))
        for dim in SAMPLE_DIMS
    }
    listing = "; ".join(f"{dim} {values[:5]}" for dim, values in missing.items() if values)
    if listing:
        raise ValueError(f"initial_conditions has no start for these coordinates of the parameter draws: {listing}.")


def _draw_conditions(
    initial_conditions: dict[str, float] | xr.DataArray | None,
    chain: int,
    draw: int,
    var_names: Sequence[str],
    predetermined: set[str],
) -> dict[str, float]:
    """
    Return one draw's initial conditions, selecting its chain and draw when the conditions carry them.

    Only predetermined values change the solution, so only they must be finite.
    """
    if _has_draws(initial_conditions):
        initial_conditions = initial_conditions.sel(chain=chain, draw=draw)
    conditions = _normalize_condition_keys(initial_conditions)

    unknown = set(conditions) - set(var_names)
    if unknown:
        raise ValueError(f"Unknown variables in initial_conditions: {unknown}. Valid: {list(var_names)}")
    non_finite = sorted(name for name in predetermined & set(conditions) if not np.isfinite(conditions[name]))
    if non_finite:
        raise ValueError(f"initial_conditions at chain {chain}, draw {draw} is not finite for: {', '.join(non_finite)}")
    return conditions


def _solve_draw(
    parameters: dict[str, float],
    conditions: dict[str, float],
    model: Model,
    problem: PerfectForesightProblem,
    shock_matrix: np.ndarray,
    steady_state_options: dict,
    solver: RootSolver,
    sparse_root_kwargs: dict,
) -> _DrawOutcome:
    """Solve one draw at its parameters, starting from its own steady state, with a fresh copy of ``solver``."""
    work_counts: dict[str, int] = dict.fromkeys(_WORK_COUNT_FIELDS, 0)
    steady_state = model.steady_state(**steady_state_options, **parameters)
    if not steady_state.success:
        return _DrawOutcome(
            path=None, steady_state_solved=False, work_counts=work_counts, failure="The steady state did not solve."
        )

    var_names = list(problem.var_names)
    steady_state_values = _ss_dict_to_array(steady_state, var_names)
    solve = partial(
        _solve_stacked,
        problem=problem,
        y_initial=_boundary_vector(conditions=conditions, ss_dict=steady_state, var_names=var_names),
        y_terminal=steady_state_values,
        shocks=shock_matrix,
        params=_build_param_matrix(param_paths=parameters, param_names=problem.param_names, model=model, T=problem.T),
        solver=copy.deepcopy(solver),
        sparse_root_kwargs=sparse_root_kwargs,
    )

    try:
        result = solve(x0=np.tile(steady_state_values, problem.T))
    except NonFiniteInitialGuessError as error:
        return _DrawOutcome(path=None, steady_state_solved=True, work_counts=work_counts, failure=str(error))

    work_counts: dict[str, int] = {field: result[field] for field in _WORK_COUNT_FIELDS}
    if not result.success:
        return _DrawOutcome(
            path=None,
            steady_state_solved=True,
            work_counts=work_counts,
            failure=f"The solver reported: {result.message}",
        )
    return _DrawOutcome(
        path=result.x.reshape(problem.T, problem.n_vars),
        steady_state_solved=True,
        work_counts=work_counts,
        failure=None,
    )


def _assemble(
    outcomes: list[_DrawOutcome],
    draws: xr.Dataset,
    problem: PerfectForesightProblem,
    model: Model,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Stack per-draw outcomes, in chain-major order, into paths and diagnostics with ``chain`` and ``draw`` dims."""
    shape = (draws.sizes["chain"], draws.sizes["draw"])
    paths = np.full((len(outcomes), problem.T, problem.n_vars), np.nan)
    success = np.zeros(len(outcomes), dtype=bool)
    steady_state_success = np.array([outcome.steady_state_solved for outcome in outcomes])
    work_counts = {field: np.zeros(len(outcomes), dtype=int) for field in _WORK_COUNT_FIELDS}
    for index, outcome in enumerate(outcomes):
        for field in _WORK_COUNT_FIELDS:
            work_counts[field][index] = outcome.work_counts[field]
        if outcome.path is not None:
            paths[index] = outcome.path
            success[index] = True

    failures = int((~success).sum())
    if failures:
        first = next(outcome.failure for outcome in outcomes if outcome.path is None)
        warnings.warn(
            f"{failures} of {success.size} draws failed to solve, and their paths are NaN. The first failure: {first}",
            stacklevel=3,
        )

    coords = {"chain": draws["chain"].values, "draw": draws["draw"].values}
    time_coords = {**coords, "time": np.arange(problem.T)}
    var_names = list(problem.var_names)
    paths = paths.reshape(*shape, problem.T, problem.n_vars)
    path_data = {
        variable.base_name: (("chain", "draw", "time"), paths[..., var_names.index(variable.base_name)])
        for variable in model.variables
    }
    diagnostics = {"success": success, "steady_state_success": steady_state_success, **work_counts}
    return (
        xr.Dataset(path_data, coords=time_coords),
        xr.Dataset(
            {name: (("chain", "draw"), values.reshape(shape)) for name, values in diagnostics.items()}, coords=coords
        ),
    )
