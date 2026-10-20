import logging

from collections.abc import Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any

import numpy as np
import xarray as xr

from gEconpy.classes.perturbation_solution import FirstOrderPerturbationSolution, SecondOrderPerturbationSolution
from gEconpy.model.model import Model
from gEconpy.model.pruning import compile_pruned_second_order_path
from gEconpy.model.statistics import build_Q_matrix
from gEconpy.model.statistics.validation import (
    _maybe_solve_model,
    _maybe_solve_steady_state,
    _validate_shock_options,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShockSpec:
    """
    Shock input used to generate an impulse response function.

    Attributes
    ----------
    mode : str
        Which of the three shock specifications is in use, one of "trajectory", "cov", or "size".
    trajectory : ndarray or None
        Path of shock values, of shape (n_periods, n_shocks). Set only when ``mode`` is "trajectory".
    cov : ndarray or None
        Shock covariance matrix, of shape (n_shocks, n_shocks). Set only when ``mode`` is "cov".
    size : float, ndarray, dict mapping str to float, or None
        Size of the shock applied to each shock in the model. Set only when ``mode`` is "size".
    orthogonalize : bool
        If True, drop the correlations in ``cov`` and draw each shock independently with its own variance.
    """

    mode: str
    trajectory: np.ndarray | None
    cov: np.ndarray | None
    size: float | np.ndarray | dict[str, float] | None
    orthogonalize: bool


def impulse_response_function(
    model: Model,
    solution: FirstOrderPerturbationSolution | None = None,
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    simulation_length: int = 40,
    shock_size: float | np.ndarray | dict[str, float] | None = None,
    shock_cov: np.ndarray | None = None,
    shock_trajectory: np.ndarray | None = None,
    return_individual_shocks: bool | None = None,
    orthogonalize_shocks: bool = False,
    random_seed: int | np.random.RandomState | None = None,
    **solve_model_kwargs,
) -> xr.Dataset:
    """
    Compute impulse response functions from the perturbation solution.

    An impulse response function traces the path of every model variable after a one-period shock hits the system
    at time zero. The shock is specified by at most one of ``shock_size``, ``shock_cov``, or ``shock_trajectory``.
    With none of them given, every shock receives a unit impulse.

    A second-order solution responds through its pruned recursion, starting from the steady state with no other
    shocks. This is the expectation of the generalized impulse response that Dynare estimates by simulation, the
    mean gap between paths with and without the impulse over random draws of the other shocks, so it does not depend
    on their covariance. Its ``level`` is the steady state plus this response, with no drift toward the stochastic
    steady state. Responses computed per shock do not add up to the response to the shocks together.

    The response comes back both as a deviation from the steady state and in levels. A log-linearized variable's
    deviation is a log deviation, so its level is ``steady_state * exp(deviation)``. Every other variable's level is
    ``steady_state + deviation``.

    Parameters
    ----------
    model : Model
        DSGE model whose variables and shocks label the output.
    solution : FirstOrderPerturbationSolution or SecondOrderPerturbationSolution, optional
        Solution from :meth:`~gEconpy.model.model.Model.solve_model`. Solved from ``model`` when omitted.
    T : ndarray, optional
        Transition matrix of the solved system. Deprecated, pass ``solution`` instead. A bare ``T`` and ``R`` carry
        no record of how the model was linearized, so their levels are computed from ``solve_model_kwargs``, which
        must describe the solve that produced them.
    R : ndarray, optional
        Selection matrix of the solved system. Deprecated, pass ``solution`` instead.
    simulation_length : int, optional
        Number of periods to compute the response over. Ignored when ``shock_trajectory`` is given, because the
        trajectory sets the length. Defaults to 40.
    shock_size : float, ndarray, or dict mapping str to float, optional
        Size of the impulse at time zero. A float applies the same impulse to every shock, an array of length
        ``n_shocks`` gives one impulse per shock in model order, and a dict gives an impulse to the named shocks
        only and restricts the output to those shocks. Defaults to None.
    shock_cov : ndarray, optional
        Covariance matrix of the shocks, of shape ``(n_shocks, n_shocks)``. The time-zero impulse is a random draw
        from this distribution. Defaults to None.
    shock_trajectory : ndarray, optional
        Path of shock values over time, of shape ``(n_periods, n_shocks)``. Defaults to None.
    return_individual_shocks : bool, optional
        If True, compute a separate response for each shock and add a ``shock`` dimension to the output. When
        omitted, responses are separated whenever the specification keeps shocks independent: a scalar, dict, or
        length ``n_shocks`` array for ``shock_size``, or a diagonal ``shock_cov``. A trajectory is never separated
        by default.
    orthogonalize_shocks : bool, optional
        Drop the correlations in ``shock_cov`` and draw each shock independently with the variance on the diagonal.
        Uncorrelated shocks are separated into per-shock responses unless ``return_individual_shocks`` says
        otherwise. Ignored for the other specifications. Defaults to False.
    random_seed : int, RandomState, or Generator, optional
        Seed for the draw taken when ``shock_cov`` is given. Defaults to None.
    **solve_model_kwargs
        Arguments forwarded to :meth:`~gEconpy.model.model.Model.solve_model`. Ignored when ``solution`` is
        provided. With ``T`` and ``R`` they describe the solve that produced the matrices, for the levels.

    Returns
    -------
    irf : Dataset
        Data variables ``deviation`` and ``level``, with dimensions ``(time, variable)``, or
        ``(shock, time, variable)`` when responses are computed per shock. The ``variable`` dimension carries the
        coordinates ``steady_state`` and ``log_linearized``.

    Examples
    --------
    A scalar ``shock_size`` applies the same impulse to every shock and separates the responses, so the result has
    a ``shock`` dimension:

    .. code-block:: python

        from gEconpy import impulse_response_function, model_from_gcn
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        irf = impulse_response_function(model, simulation_length=20, shock_size=0.01, verbose=False)
        print(irf["deviation"].sel(shock="epsilon_A", variable="Y").values[:5])
        print(irf["level"].sel(shock="epsilon_A", variable="Y").values[:5])
    """
    rng = np.random.default_rng(random_seed)

    if T is not None or R is not None or isinstance(solution, np.ndarray):
        # Called from here so the deprecation warning points at the caller's line.
        T, R = _maybe_solve_model(model, solution, T, R, **solve_model_kwargs)
        if T is None or R is None:
            raise _unsolved_model_error("compute a response from")
        solution = _wrap_policy_matrices(model, T=T, R=R, **solve_model_kwargs)
    elif solution is None:
        solution = _solve_or_raise(model, purpose="compute a response from", **solve_model_kwargs)

    spec = _make_shock_spec(shock_size, shock_cov, shock_trajectory, orthogonalize_shocks)

    variable_names = [x.base_name for x in model.variables]
    shock_names = [x.base_name for x in model.shocks]
    selected_shock_names = _get_selected_shock_names(spec, shock_names)

    n_shocks = len(shock_names)
    n_selected_shocks = len(selected_shock_names)
    selected_shock_idxs = [i for i, name in enumerate(shock_names) if name in selected_shock_names]

    if spec.mode == "trajectory":
        simulation_length = spec.trajectory.shape[0]

    apply_shocks_individually = _infer_shocks_are_individual(return_individual_shocks, spec, n_selected_shocks)

    if not apply_shocks_individually:
        trajectory = _build_trajectory(spec, simulation_length, n_shocks, selected_shock_names, rng)
        responses = _impulse_response(solution, trajectory)
        return _irf_to_dataset(responses, solution=solution, variable_names=variable_names, shock_names=None)

    full_trajectory = _build_trajectory(spec, simulation_length, n_shocks, shock_names, rng)
    responses = np.zeros((n_selected_shocks, simulation_length, len(variable_names)), dtype=float)
    for i, shock_idx in enumerate(selected_shock_idxs):
        single_shock_trajectory = np.zeros_like(full_trajectory)
        single_shock_trajectory[:, shock_idx] = full_trajectory[:, shock_idx]
        responses[i] = _impulse_response(solution, single_shock_trajectory)

    return _irf_to_dataset(
        responses, solution=solution, variable_names=variable_names, shock_names=selected_shock_names
    )


def simulate(
    model: Model,
    solution: FirstOrderPerturbationSolution | None = None,
    T: np.ndarray | None = None,
    R: np.ndarray | None = None,
    n_simulations: int = 1,
    simulation_length: int = 40,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
    random_seed: int | np.random.RandomState | None = None,
    **solve_model_kwargs,
) -> xr.DataArray:
    """
    Simulate trajectories of the perturbation solution driven by Gaussian shocks.

    The shock covariance comes from exactly one of ``shock_std_dict``, ``shock_cov_matrix``, or ``shock_std``. A
    second-order solution is simulated with pruning, and its risk correction is evaluated at that covariance.

    Parameters
    ----------
    model : Model
        DSGE model whose variables and shocks label the output.
    solution : FirstOrderPerturbationSolution or SecondOrderPerturbationSolution, optional
        Solution from :meth:`~gEconpy.model.model.Model.solve_model`. Solved from ``model`` when omitted.
    T : ndarray, optional
        Transition matrix of the solved system. Deprecated, pass ``solution`` instead.
    R : ndarray, optional
        Selection matrix of the solved system. Deprecated, pass ``solution`` instead.
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
    **solve_model_kwargs
        Arguments forwarded to :meth:`~gEconpy.model.model.Model.solve_model`. Ignored when ``solution``, or ``T``
        and ``R``, are provided.

    Returns
    -------
    simulations : DataArray
        Simulated deviations from the steady state, with dimensions ``(simulation, time, variable)``.

    Examples
    --------
    A scalar ``shock_std`` gives every shock the same standard deviation:

    .. code-block:: python

        from gEconpy import model_from_gcn, simulate
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        paths = simulate(model, n_simulations=5, simulation_length=100, shock_std=0.01, random_seed=0, verbose=False)
        print(paths.sel(variable="Y").std(dim="time").values)
    """
    epsilons = _draw_shocks(
        model=model,
        n_simulations=n_simulations,
        simulation_length=simulation_length,
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        random_seed=random_seed,
    )

    if isinstance(solution, FirstOrderPerturbationSolution) and (T is not None or R is not None):
        raise ValueError("Pass either solution or T and R, not both.")
    if solution is None and T is None and R is None:
        solution = _solve_or_raise(model, purpose="simulate", **solve_model_kwargs)

    if isinstance(solution, SecondOrderPerturbationSolution):
        shock_cov = build_Q_matrix(
            model_shocks=model.shocks,
            shock_std_dict=shock_std_dict,
            shock_cov_matrix=shock_cov_matrix,
            shock_std=shock_std,
        )
        trajectories = _simulate_pruned_system(solution, epsilons, ghs2=solution.ghs2(shock_cov))
    else:
        T, R = _maybe_solve_model(model, solution, T, R, **solve_model_kwargs)
        trajectories = _simulate_linear_system(T, R, epsilons)

    return xr.DataArray(
        trajectories,
        dims=["simulation", "time", "variable"],
        coords={
            "variable": [x.base_name for x in model.variables],
            "simulation": np.arange(n_simulations),
            "time": np.arange(simulation_length),
        },
    )


def _wrap_policy_matrices(
    model: Model, T: np.ndarray, R: np.ndarray, **solve_model_kwargs
) -> FirstOrderPerturbationSolution:
    """
    Wrap bare policy matrices with the steady state and log-linearization that ``solve_model_kwargs`` describe.

    The arguments are read as :meth:`~gEconpy.model.model.Model.solve_model` reads them, so matrices solved with the
    same arguments get the levels that solve would have reported.
    """
    named = {
        name
        for name, parameter in signature(model.solve_model).parameters.items()
        if parameter.kind is not Parameter.VAR_KEYWORD
    }
    parameter_updates = {name: value for name, value in solve_model_kwargs.items() if name not in named}
    steady_state_kwargs = {"verbose": False, "progressbar": False} | (
        solve_model_kwargs.get("steady_state_kwargs") or {}
    )
    steady_state = _maybe_solve_steady_state(
        model,
        steady_state=solve_model_kwargs.get("steady_state"),
        steady_state_kwargs=steady_state_kwargs,
        parameter_updates=parameter_updates,
    )

    return model._perturbation_solution(
        T=T,
        R=R,
        steady_state=steady_state.to_string(),
        log_linearize=solve_model_kwargs.get("log_linearize", True),
        not_loglin_variables=solve_model_kwargs.get("not_loglin_variables"),
        loglin_negative_ss=solve_model_kwargs.get("loglin_negative_ss", False),
    )


def _solve_or_raise(model: Model, purpose: str, **solve_model_kwargs) -> FirstOrderPerturbationSolution:
    """Solve ``model``, raising when the solve fails under ``on_failure="ignore"``."""
    solved = model.solve_model(**{**solve_model_kwargs, "return_matrices": False})
    if not isinstance(solved, FirstOrderPerturbationSolution):
        raise _unsolved_model_error(purpose)
    return solved


def _unsolved_model_error(purpose: str) -> ValueError:
    return ValueError(f"The model could not be solved, so there is no policy function to {purpose}.")


def _draw_shocks(
    model: Model,
    n_simulations: int,
    simulation_length: int,
    shock_std_dict: dict[str, float] | None = None,
    shock_cov_matrix: np.ndarray | None = None,
    shock_std: np.ndarray | list | float | None = None,
    random_seed: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Draw ``(n_simulations, simulation_length, n_shocks)`` Gaussian shocks from :func:`simulate`'s arguments."""
    rng = np.random.default_rng(random_seed)
    shocks = model.shocks

    _validate_shock_options(
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
        shocks=shocks,
    )

    Q = build_Q_matrix(
        model_shocks=shocks,
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        shock_std=shock_std,
    )

    return rng.multivariate_normal(
        mean=np.zeros(len(shocks)),
        cov=Q,
        size=(n_simulations, simulation_length),
        method="svd",
    )


def _validate_irf_shock_arguments(*values_with_names: tuple[str, Any]) -> None:
    provided_names = [name for name, value in values_with_names if value is not None]
    if len(provided_names) > 1:
        raise ValueError(f"Only one of {', '.join(provided_names)} may be specified, got {len(provided_names)}.")


def _is_diagonal(M: np.ndarray) -> bool:
    return np.allclose(M, np.diag(np.diag(M)))


def _get_selected_shock_names(spec: ShockSpec, shock_names: list[str]) -> list[str]:
    """Restrict the shock list to the keys of a ``shock_size`` dict, in model order."""
    if spec.mode != "size" or not isinstance(spec.size, dict):
        return shock_names

    if len(spec.size) == 0:
        raise ValueError("Shock size cannot be empty.")

    unknown_shocks = set(spec.size) - set(shock_names)
    if unknown_shocks:
        raise ValueError(f"shock_size dict contains unknown shock names: {unknown_shocks}")

    return [name for name in shock_names if name in spec.size]


def _infer_shocks_are_individual(
    requested: bool | None,
    shock_spec: ShockSpec,
    n_shocks: int,
) -> bool:
    if requested is not None:
        return requested

    if shock_spec.mode == "size":
        if isinstance(shock_spec.size, int | float | dict):
            return True

        size = np.asarray(shock_spec.size)
        return size.ndim == 0 or size.shape == (n_shocks,)

    if shock_spec.mode == "cov":
        return shock_spec.orthogonalize or _is_diagonal(np.asarray(shock_spec.cov))

    return False


def _make_shock_spec(
    shock_size: float | np.ndarray | dict[str, float] | None,
    shock_cov: np.ndarray | None,
    shock_trajectory: np.ndarray | None,
    orthogonalize_shocks: bool,
) -> ShockSpec:
    _validate_irf_shock_arguments(
        ("shock_size", shock_size),
        ("shock_cov", shock_cov),
        ("shock_trajectory", shock_trajectory),
    )

    if shock_trajectory is not None:
        mode = "trajectory"
    elif shock_cov is not None:
        mode = "cov"
    else:
        mode = "size"

    return ShockSpec(
        mode=mode, trajectory=shock_trajectory, cov=shock_cov, size=shock_size, orthogonalize=orthogonalize_shocks
    )


def _shock_vector_from_spec(
    size: float | np.ndarray | dict[str, float] | None,
    shock_names: Sequence[str],
) -> np.ndarray:
    """Return the ``(n_shocks,)`` impulse applied at time zero."""
    n_shocks = len(shock_names)
    if size is None:
        return np.ones(n_shocks)
    if isinstance(size, int | float):
        return np.full(n_shocks, float(size))
    if isinstance(size, dict):
        return np.array([float(size.get(name, 0.0)) for name in shock_names], dtype=float)

    impulse = np.asarray(size, dtype=float)
    if impulse.shape != (n_shocks,):
        raise ValueError(f"shock_size array must have shape ({n_shocks},); got {impulse.shape}.")
    return impulse


def _build_trajectory(
    spec: ShockSpec,
    simulation_length: int,
    n_shocks: int,
    shock_names: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Convert a :class:`ShockSpec` into a ``(simulation_length, n_shocks)`` shock trajectory."""
    match spec.mode:
        case "trajectory":
            trajectory = np.asarray(spec.trajectory, dtype=float)
            if trajectory.ndim != 2 or trajectory.shape[1] != n_shocks:
                raise ValueError(f"shock_trajectory must have shape (T, {n_shocks}); got {trajectory.shape}.")

        case "cov":
            trajectory = np.zeros((simulation_length, n_shocks), dtype=float)
            Q = np.asarray(spec.cov, dtype=float)
            if Q.shape != (n_shocks, n_shocks):
                raise ValueError(f"shock_cov must be ({n_shocks}, {n_shocks}); got {Q.shape}.")
            standard_draw = rng.standard_normal(n_shocks)
            if spec.orthogonalize:
                trajectory[0] = np.sqrt(np.diag(Q)) * standard_draw
            else:
                trajectory[0] = np.linalg.cholesky(Q) @ standard_draw

        case "size":
            trajectory = np.zeros((simulation_length, n_shocks), dtype=float)
            trajectory[0] = _shock_vector_from_spec(spec.size, shock_names)

        case _:
            raise RuntimeError(f"Unexpected ShockSpec mode: {spec.mode}. You shouldn't get here, please report a bug.")

    return trajectory


def _simulate_linear_system(T: np.ndarray, R: np.ndarray, shock_trajectory: np.ndarray) -> np.ndarray:
    """
    Iterate :math:`x_t = T x_{t-1} + R e_t` from :math:`x_{-1} = 0` along the shock path :math:`e_t`.

    ``shock_trajectory`` has shape ``(..., n_periods, n_shocks)`` and any leading dimensions are simulated
    independently, giving states of shape ``(..., n_periods, n_vars)``.
    """
    T = np.asarray(T)
    R = np.asarray(R)
    n_periods = shock_trajectory.shape[-2]
    n_vars = T.shape[0]

    states = np.zeros((*shock_trajectory.shape[:-2], n_periods, n_vars), dtype=float)
    states[..., 0, :] = shock_trajectory[..., 0, :] @ R.T
    for t in range(1, n_periods):
        states[..., t, :] = states[..., t - 1, :] @ T.T + shock_trajectory[..., t, :] @ R.T
    return states


def _impulse_response(solution: FirstOrderPerturbationSolution, shock_trajectory: np.ndarray) -> np.ndarray:
    """Respond to ``shock_trajectory`` from the steady state, in model order."""
    if isinstance(solution, SecondOrderPerturbationSolution):
        # The risk correction moves the paths with and without the impulse alike, so it drops out of the response.
        return _simulate_pruned_system(solution, shock_trajectory, ghs2=np.zeros(len(solution.variable_names)))
    return _simulate_linear_system(solution.T, solution.R, shock_trajectory)


def _simulate_pruned_system(
    solution: SecondOrderPerturbationSolution, shock_trajectory: np.ndarray, ghs2: np.ndarray
) -> np.ndarray:
    """
    Simulate the pruned second-order solution from the steady state along the shock path.

    ``shock_trajectory`` has shape ``(..., n_periods, n_shocks)`` and any leading dimensions are simulated
    independently, giving deviations of shape ``(..., n_periods, n_vars)`` in model order.
    """
    shock_trajectory = np.asarray(shock_trajectory, dtype=float)
    batch_shape = shock_trajectory.shape[:-2]

    deviations = compile_pruned_second_order_path()(
        ghx=solution.ghx,
        ghu=solution.ghu,
        ghxx=solution.ghxx,
        ghxu=solution.ghxu,
        ghuu=solution.ghuu,
        ghs2=ghs2,
        shocks=shock_trajectory.reshape(-1, *shock_trajectory.shape[-2:]),
        state_positions=solution.dr_order.lag_positions,
    )
    deviations = deviations.reshape(*batch_shape, *deviations.shape[-2:])
    return deviations[..., solution.dr_order.inv_var_order]


def _irf_to_dataset(
    responses: np.ndarray,
    solution: FirstOrderPerturbationSolution,
    variable_names: list[str],
    shock_names: list[str] | None,
) -> xr.Dataset:
    """Package deviations and their levels together, with the steady state and log flags on ``variable``."""
    # A single Dataset call is several times faster than building DataArrays and merging them.
    dims = ("time", "variable") if shock_names is None else ("shock", "time", "variable")
    coords = {} if shock_names is None else {"shock": list(shock_names)}
    coords |= {
        "time": np.arange(responses.shape[-2]),
        "variable": list(variable_names),
        "steady_state": ("variable", solution.steady_state),
        "log_linearized": ("variable", solution.log_linearized),
    }

    return xr.Dataset(
        {"deviation": (dims, responses), "level": (dims, solution.to_levels(responses))},
        coords=coords,
    )
