from __future__ import annotations

import logging
import multiprocessing
import warnings

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import cloudpickle
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import xarray as xr

from rich.progress import Progress
from scipy.linalg import LinAlgError

from gEconpy.exceptions import PerturbationSolutionNotFoundException
from gEconpy.model.perturbation import (
    check_bk_condition as _check_bk_condition,
)
from gEconpy.model.perturbation import (
    compute_bk_eigenvalues_pt,
    residual_norms,
    statespace_to_gEcon_representation,
)
from gEconpy.model.sampling import (
    sample_from_priors,
    sample_from_priors_qmc,
    sample_uniform_from_priors,
)
from gEconpy.model.statistics.validation import _maybe_linearize_model
from gEconpy.pytensorf.compile import rewrite_pregrad
from gEconpy.solvers.backward_looking import solve_policy_function_with_backward_direct
from gEconpy.solvers.cycle_reduction import solve_policy_function_with_cycle_reduction
from gEconpy.solvers.gensys import solve_policy_function_with_gensys

if TYPE_CHECKING:
    from gEconpy.model.model import Model

_log = logging.getLogger(__name__)

_SHARED: dict = {"model": None, "kwargs": {}}


def _init_worker(model_or_bytes, kwargs: dict, use_pickle: bool = False) -> None:
    if use_pickle:
        _SHARED["model"] = cloudpickle.loads(model_or_bytes)
    else:
        _SHARED["model"] = model_or_bytes
    _SHARED["kwargs"] = kwargs


def _worker_fn(updates: dict):
    return _check_one_draw(_SHARED["model"], updates, **_SHARED["kwargs"])


def _pick_start_method() -> str:
    available = multiprocessing.get_all_start_methods()
    if "fork" in available:
        return "fork"
    if "forkserver" in available:
        return "forkserver"
    return "spawn"


def _solve_perturbation(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    solver: str,
    n_variables: int,
    tol: float,
    max_iter: int,
    backward_looking: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Dispatch to the appropriate perturbation solver and return (T, R).

    Returns ``(None, None)`` when the solver fails or reports non-convergence.
    """
    effective_solver = "backward_direct" if backward_looking else solver

    if effective_solver == "cycle_reduction":
        T, R, _result, _log_norm = solve_policy_function_with_cycle_reduction(A, B, C, D, max_iter, tol, False)
        return (T, R) if T is not None else (None, None)

    if effective_solver == "gensys":
        G_1, _const, impact, _f_mat, _f_wt, _y_wt, _gev, eu, _loose = solve_policy_function_with_gensys(A, B, C, D, tol)
        if not all(x == 1 for x in eu[:2]):
            return None, None
        return G_1[:n_variables, :n_variables], impact[:n_variables, :]

    if effective_solver == "backward_direct":
        return solve_policy_function_with_backward_direct(A, B, C, D)

    raise ValueError(f"Unknown solver {solver!r}")


_NUMERICAL_ERRORS = (ValueError, ArithmeticError, LinAlgError, RuntimeError)


def _check_one_draw(
    model: Model,
    updates: dict,
    solver: str,
    steady_state_kwargs: dict,
    linearize_kwargs: dict,
    tol: float,
    max_iter: int,
    norm_tol: float,
) -> tuple[str | None, float, float]:
    """Run the full solvability pipeline for one parameter draw.

    Returns
    -------
    tuple of (failure_step, norm_deterministic, norm_stochastic)
        ``failure_step`` is ``None`` on success, or the name of the failing stage.
        Norms are ``nan`` when not reached.
    """
    failure_step: str | None = None
    deterministic_norm = np.nan
    stochastic_norm = np.nan

    ss = T = R = None
    try:
        ss = model.steady_state(verbose=False, progressbar=False, **steady_state_kwargs, **updates)
        if not ss.success:
            failure_step = "steady_state"
    except _NUMERICAL_ERRORS:
        failure_step = "steady_state"

    A = B = C = D = None
    if failure_step is None:
        try:
            A, B, C, D = model.linearize_model(steady_state=ss, verbose=False, **linearize_kwargs, **updates)
            T, R = _solve_perturbation(A, B, C, D, solver, model.n_variables, tol, max_iter, model._backward_looking)
            if T is None:
                failure_step = "perturbation"
        except _NUMERICAL_ERRORS:
            failure_step = "perturbation"

    if failure_step is None and not bool(_check_bk_condition(A, B, C, D, verbose=False, return_value="bool")):
        failure_step = "blanchard-kahn"

    if failure_step is None:
        try:
            P, Q, _, _, A_prime, R_prime, S_prime = statespace_to_gEcon_representation(A, T, R, tol)
            deterministic_norm, stochastic_norm = residual_norms(B, C, D, Q, P, A_prime, R_prime, S_prime)
        except _NUMERICAL_ERRORS:
            failure_step = "deterministic_norm"

    if failure_step is None:
        if deterministic_norm > norm_tol:
            failure_step = "deterministic_norm"
        elif stochastic_norm > norm_tol:
            failure_step = "stochastic_norm"

    return failure_step, deterministic_norm, stochastic_norm


def summarize_perturbation_solution(
    linear_system: Sequence[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    perturbation_solution: Sequence[np.ndarray | None, np.ndarray | None],
    model: Model,
):
    A, B, C, D = linear_system
    T, R = perturbation_solution
    if T is None or R is None:
        raise PerturbationSolutionNotFoundException()

    coords = {
        "equation": np.arange(A.shape[0]).astype(int),
        "variable": [x.base_name for x in model.variables],
        "shock": [x.base_name for x in model.shocks],
    }

    return xr.Dataset(
        data_vars={
            "A": (("equation", "variable"), A),
            "B": (("equation", "variable"), B),
            "C": (("equation", "variable"), C),
            "D": (("equation", "shock"), D),
            "T": (("equation", "variable"), T),
            "R": (("equation", "shock"), R),
        },
        coords=coords,
    )


def check_bk_condition(
    model: Model,
    *,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None,
    C: np.ndarray | None = None,
    D: np.ndarray | None = None,
    tol=1e-8,
    on_failure: Literal["raise", "ignore"] = "ignore",
    return_value: Literal["dataframe", "bool", None] = "dataframe",
    **linearize_model_kwargs,
) -> bool | pd.DataFrame | None:
    """
    Compute the generalized eigenvalues of system in the form presented in [1].

    Per [2], the number of unstable eigenvalues (:math:`|v| > 1`) should not be greater than the number of
    forward-looking variables. Failing this test suggests timing problems in the definition of the model.

    Parameters
    ----------
    model: Model
        DSGE model.
    A, B, C, D : np.ndarray, optional
        Jacobian matrices. If not all provided, ``model.linearize_model`` is called.
    tol : float
        Tolerance for zero.
    on_failure : str
        ``'raise'`` or ``'ignore'``.
    return_value : str or None
        ``'dataframe'``, ``'bool'``, or None.
    **linearize_model_kwargs
        Forwarded to ``model.linearize_model``.

    Returns
    -------
    bk_result : bool, DataFrame, or None
    """
    verbose = linearize_model_kwargs.get("verbose", True)
    A, B, C, D = _maybe_linearize_model(model, A, B, C, D, **linearize_model_kwargs)
    return _check_bk_condition(
        A,
        B,
        C,
        D,
        tol=tol,
        verbose=verbose,
        on_failure=on_failure,
        return_value=return_value,
    )


def eigenvalue_sensitivity(
    model: Model,
    *,
    verbose: bool = True,
    steady_state: dict | None = None,
    steady_state_kwargs: dict | None = None,
    **parameter_updates,
) -> xr.Dataset:
    r"""
    Compute the sensitivity of system eigenvalues to model parameters.

    For each eigenvalue of the linearized DSGE system's augmented form, computes the derivative
    of the real and imaginary parts with respect to every free parameter. This reveals which
    parameters push eigenvalues toward or away from the unit circle, potentially moving the
    model in or out of Blanchard-Kahn stability.

    Parameters
    ----------
    model : Model
        A gEconpy DSGE model.
    verbose : bool, default True
        Forwarded to steady-state and linearization routines.
    steady_state : dict, optional
        Pre-computed steady state. If not provided, solved internally.
    steady_state_kwargs : dict, optional
        Keyword arguments forwarded to ``model.steady_state``.
    **parameter_updates
        Parameter overrides (e.g. ``beta=0.98``).

    Returns
    -------
    xr.Dataset
        Dataset with two data variables:

        - ``eigenvalues`` : shape ``(eigenvalue, component)`` where ``component`` is
          ``[real, imaginary, modulus]``.
        - ``gradients`` : shape ``(eigenvalue, part, parameter)`` where ``part`` is
          ``[real, imaginary]``.

    Notes
    -----
    Eigenvalues are computed from the Sims (2002) augmented system via the ``real_eig`` Op,
    which provides exact reverse-mode gradients.

    Eigenvalues are sorted by modulus (ascending), so zeros appear first and large/infinite
    eigenvalues appear last.

    Examples
    --------
    .. code-block:: python

        from gEconpy.model.build import model_from_gcn
        from gEconpy.model.statistics import eigenvalue_sensitivity

        model = model_from_gcn("rbc.gcn")
        ds = eigenvalue_sensitivity(model, verbose=False)

        # Filter to finite eigenvalues if desired
        mod = ds.eigenvalues.sel(component="modulus")
        finite_mask = (mod > 1e-6) & (mod < 1e6)
        finite_idx = ds.eigenvalue.values[finite_mask.values]
        finite = ds.sel(eigenvalue=finite_idx)
    """
    if steady_state_kwargs is None:
        steady_state_kwargs = {}

    param_dict = model.parameters(**parameter_updates)
    if steady_state is None:
        steady_state = model.steady_state(**param_dict, verbose=verbose, **steady_state_kwargs)

    jacobians, ss_nodes, param_nodes = model.symbolic_linearization(steady_state=steady_state, verbose=False)
    A_sym, B_sym, C_sym, D_sym = jacobians
    param_names = [p.name for p in param_nodes]

    lead_var_idx = model.lead_var_idx
    eigvals_re_pt, eigvals_im_pt = compute_bk_eigenvalues_pt(A_sym, B_sym, C_sym, D_sym, lead_var_idx)
    eigvals_re_pt = rewrite_pregrad(eigvals_re_pt)
    eigvals_im_pt = rewrite_pregrad(eigvals_im_pt)

    n_eig = model.n_variables + model.n_forward

    jac_re = pt.stack(pt.jacobian(eigvals_re_pt, param_nodes), axis=1)
    jac_im = pt.stack(pt.jacobian(eigvals_im_pt, param_nodes), axis=1)

    ss_values = {k.removesuffix("_ss"): v for k, v in steady_state.items()}

    all_inputs = list(ss_nodes) + list(param_nodes)
    input_vals = [float(ss_values[v.base_name]) for v in model.variables]
    input_vals += [float(param_dict[n.name]) for n in param_nodes]

    f = pytensor.function(
        all_inputs, [eigvals_re_pt, eigvals_im_pt, jac_re, jac_im], on_unused_input="ignore", mode=model._mode
    )
    re_vals, im_vals, jac_re_vals, jac_im_vals = f(*input_vals)
    mod_vals = np.sqrt(re_vals**2 + im_vals**2)

    eigenvalue_coords = np.arange(n_eig)
    eigenvalues_data = np.stack([re_vals, im_vals, mod_vals], axis=1)
    gradients_data = np.stack([jac_re_vals, jac_im_vals], axis=1)

    return xr.Dataset(
        {
            "eigenvalues": (["eigenvalue", "component"], eigenvalues_data),
            "gradients": (["eigenvalue", "part", "parameter"], gradients_data),
        },
        coords={
            "eigenvalue": eigenvalue_coords,
            "component": ["real", "imaginary", "modulus"],
            "part": ["real", "imaginary"],
            "parameter": param_names,
        },
    )


def solvability_check(
    model: Model,
    samples: pd.DataFrame,
    *,
    cores: int = 1,
    solver: str = "cycle_reduction",
    steady_state_kwargs: dict | None = None,
    linearize_kwargs: dict | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
    norm_tol: float = 1e-8,
    progressbar: bool = True,
) -> pd.DataFrame:
    """Check whether each row of ``samples`` yields a solvable DSGE model.

    Each row is pushed through the full solution pipeline:
    steady state → linearization → perturbation solve → Blanchard-Kahn check →
    residual norms. The first step that fails determines the ``failure_step`` label.

    Parameters
    ----------
    model : Model
        A compiled DSGE model. Must have steady-state and linearization
        functions compiled (i.e. ``model.steady_state()`` must have been called
        at least once before passing to this function).
    samples : pd.DataFrame
        Parameter draws. Column names must be a subset of the model's parameter
        names. Unspecified parameters use the model's calibrated defaults.
    cores : int, default 1
        Number of parallel worker processes. Uses ``fork`` on macOS/Linux for
        near-linear speedup with zero serialization overhead. Falls back to
        ``spawn`` on Windows (slower; ~12 s pool startup overhead).
    solver : str, default ``"cycle_reduction"``
        Perturbation solver. One of ``"cycle_reduction"``, ``"gensys"``.
        Backward-looking models always use ``"backward_direct"`` regardless of
        this setting.
    steady_state_kwargs : dict, optional
        Extra keyword arguments forwarded to ``model.steady_state()``.
    linearize_kwargs : dict, optional
        Extra keyword arguments forwarded to ``model.linearize_model()``.
    tol : float, default 1e-8
        Solver convergence tolerance.
    max_iter : int, default 100
        Maximum solver iterations.
    norm_tol : float, default 1e-8
        Threshold for deterministic and stochastic residual norms.
    progressbar : bool, default True
        Show a ``rich`` progress bar (serial mode only).

    Returns
    -------
    pd.DataFrame
        The input ``samples`` with three additional columns:

        - ``failure_step`` : str or None. ``None`` on success; otherwise the
          name of the first failing stage: ``"steady_state"``,
          ``"perturbation"``, ``"blanchard-kahn"``, ``"deterministic_norm"``,
          or ``"stochastic_norm"``.
        - ``norm_deterministic`` : float. Deterministic residual norm, or
          ``nan`` if not reached.
        - ``norm_stochastic`` : float. Stochastic residual norm, or ``nan``
          if not reached.
    """
    ss_kwargs = steady_state_kwargs or {}
    lin_kwargs = linearize_kwargs or {}

    shared_kwargs = {
        "solver": solver,
        "steady_state_kwargs": ss_kwargs,
        "linearize_kwargs": lin_kwargs,
        "tol": tol,
        "max_iter": max_iter,
        "norm_tol": norm_tol,
    }

    param_dicts = [{k: v for k, v in row._asdict().items() if k != "Index"} for row in samples.itertuples()]

    if cores == 1:
        results = _run_serial(model, param_dicts, shared_kwargs, progressbar)
    else:
        results = _run_parallel(model, param_dicts, shared_kwargs, cores)

    failure_steps, norms_det, norms_stoch = zip(*results, strict=False)

    out = samples.copy()
    out["failure_step"] = list(failure_steps)
    out["norm_deterministic"] = list(norms_det)
    out["norm_stochastic"] = list(norms_stoch)
    return out


def _run_serial(
    model: Model,
    param_dicts: list[dict],
    shared_kwargs: dict,
    progressbar: bool,
) -> list[tuple]:
    if progressbar:
        with Progress() as progress:
            task = progress.add_task("Checking solvability...", total=len(param_dicts))
            results = []
            for updates in param_dicts:
                results.append(_check_one_draw(model, updates, **shared_kwargs))
                progress.advance(task)
        return results
    return [_check_one_draw(model, updates, **shared_kwargs) for updates in param_dicts]


def _run_parallel(
    model: Model,
    param_dicts: list[dict],
    shared_kwargs: dict,
    cores: int,
) -> list[tuple]:
    method = _pick_start_method()
    use_pickle = method != "fork"

    model_payload = cloudpickle.dumps(model, protocol=-1) if use_pickle else model

    mp_ctx = multiprocessing.get_context(method)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*multi-threaded.*fork.*")
        with mp_ctx.Pool(
            cores,
            initializer=_init_worker,
            initargs=(model_payload, shared_kwargs, use_pickle),
        ) as pool:
            results = list(pool.imap_unordered(_worker_fn, param_dicts))
    return results


def _collect_priors(
    model: Model,
    param_subset: list[str] | None = None,
) -> dict:
    """Merge model parameter and (optionally) shock priors, then filter.

    Returns
    -------
    dict
        Prior distributions keyed by parameter name.

    Raises
    ------
    ValueError
        If no priors exist or ``param_subset`` contains unknown names.
    """
    priors = dict(model.param_priors)

    if not priors:
        raise ValueError(
            "Model has no param_priors defined. Use solvability_check with a "
            "manually constructed samples DataFrame instead."
        )

    if param_subset is not None:
        unknown = set(param_subset) - set(priors)
        if unknown:
            raise ValueError(f"param_subset contains names not found in model.param_priors: {unknown}")
        priors = {k: v for k, v in priors.items() if k in param_subset}

    return priors


def prior_solvability_check(
    model: Model,
    n_samples: int,
    *,
    seed: int | np.random.Generator | None = None,
    param_subset: list[str] | None = None,
    method: str = "lhs",
    hdi_prob: float = 0.99,
    **kwargs,
) -> pd.DataFrame:
    """Sample from the model's preliz priors and check solvability.

    Thin wrapper: draws a parameter ``DataFrame`` from the model's prior
    distributions, then delegates to :func:`solvability_check`.

    Parameters
    ----------
    model : Model
        Must have at least one prior defined via ``param_priors``.
    n_samples : int
        Number of parameter draws.
    seed : int or Generator, optional
        Random seed.
    param_subset : list of str, optional
        If given, only these parameters are sampled; others use model defaults.
    method : str, default ``"lhs"``
        Sampling strategy:

        - ``"random"`` — Monte Carlo via ``.rvs()``.
        - ``"lhs"``, ``"sobol"``, ``"halton"``, ``"poisson_disk"`` — uniform
          QMC over HDI bounds (recommended).
        - ``"sobol_ppf"``, ``"halton_ppf"`` — QMC via inverse-CDF.
    hdi_prob : float, default 0.99
        HDI probability for bound computation. Ignored for ``"random"`` and
        ``"*_ppf"`` methods.
    **kwargs
        Forwarded to :func:`solvability_check`.

    Returns
    -------
    pd.DataFrame
        See :func:`solvability_check` return value.
    """
    priors = _collect_priors(model, param_subset)

    if method == "random":
        samples = sample_from_priors(priors, n_samples, seed=seed)
    elif method.endswith("_ppf"):
        qmc_method = method.removesuffix("_ppf")
        samples = sample_from_priors_qmc(priors, n_samples, seed=seed, method=qmc_method)
    else:
        samples = sample_uniform_from_priors(priors, n_samples, seed=seed, method=method, hdi_prob=hdi_prob)

    return solvability_check(model, samples, **kwargs)
