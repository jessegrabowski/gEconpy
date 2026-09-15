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

# Worker processes read the model and solver settings from this module-level dict. Under ``fork`` the initializer
# stores the live model, so nothing is serialized. Under ``spawn`` it stores a cloudpickle payload that is unpickled
# once per worker.
_SHARED: dict = {"model": None, "kwargs": {}}

_NUMERICAL_ERRORS = (ValueError, ArithmeticError, LinAlgError, RuntimeError)


def summarize_perturbation_solution(
    linear_system: Sequence[np.ndarray],
    perturbation_solution: Sequence[np.ndarray | None],
    model: "Model",
) -> xr.Dataset:
    """
    Collect the linearized system and its perturbation solution into a labeled dataset.

    Parameters
    ----------
    linear_system : sequence of ndarray
        The four Jacobian matrices A, B, C and D of the linearized model.
    perturbation_solution : sequence of ndarray
        The transition matrix T and the selection matrix R. Either being None raises
        :class:`~gEconpy.exceptions.PerturbationSolutionNotFoundException`.
    model : Model
        Model the solution belongs to, used to label the variable and shock coordinates.

    Returns
    -------
    summary : Dataset
        Dataset holding A, B, C, D, T and R on shared equation, variable, and shock coordinates.

    Examples
    --------
    Gather the linearization and its solution into one object whose entries can be selected by name:

    .. code-block:: python

        from gEconpy import model_from_gcn, summarize_perturbation_solution
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        linear_system = model.linearize_model(verbose=False)
        policy_function = model.solve_model(verbose=False)

        summary = summarize_perturbation_solution(linear_system, policy_function, model)
        technology_response = summary["R"].sel(shock="epsilon_A")
    """
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
    model: "Model",
    *,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None,
    C: np.ndarray | None = None,
    D: np.ndarray | None = None,
    tol: float = 1e-8,
    on_failure: Literal["raise", "ignore"] = "ignore",
    return_value: Literal["dataframe", "bool", None] = "dataframe",
    **linearize_model_kwargs,
) -> bool | pd.DataFrame | None:
    """
    Check the Blanchard-Kahn condition of a model.

    Computes the generalized eigenvalues of the linearized system in the Sims (2002) form. Per Blanchard and Kahn
    (1980), a unique stable solution exists when the number of unstable eigenvalues (modulus greater than one) equals
    the number of forward-looking variables. Failing this test points to timing problems in the model definition.

    Parameters
    ----------
    model : Model
        DSGE model.
    A, B, C, D : ndarray, optional
        Jacobian matrices of the linearized system. ``model.linearize_model`` is called unless all four are given.
        Defaults to None.
    tol : float, optional
        Threshold below which numerical values count as zero. Defaults to 1e-8.
    on_failure : str, optional
        One of ``'raise'`` or ``'ignore'``. Action to take when the condition is not satisfied. Defaults to
        ``'ignore'``.
    return_value : str or None, optional
        One of ``'dataframe'``, ``'bool'``, or None. Selects what to return. Defaults to ``'dataframe'``.
    **linearize_model_kwargs
        Forwarded to ``model.linearize_model``. ``verbose`` also controls whether the result is logged.

    Returns
    -------
    result : DataFrame, bool, or None
        With ``'dataframe'``, a DataFrame with columns ``Modulus``, ``Real``, and ``Imaginary``. With ``'bool'``, True
        when the Blanchard-Kahn condition is satisfied. With None, nothing.

    Examples
    --------
    Inspect the eigenvalues of a model and count how many lie outside the unit circle:

    .. code-block:: python

        from gEconpy import check_bk_condition, model_from_gcn
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        eigenvalues = check_bk_condition(model, verbose=False)
        n_unstable = (eigenvalues["Modulus"] > 1).sum()
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
    model: "Model",
    *,
    verbose: bool = True,
    steady_state: dict | None = None,
    steady_state_kwargs: dict | None = None,
    **parameter_updates,
) -> xr.Dataset:
    r"""
    Compute the sensitivity of the system eigenvalues to the model parameters.

    For each eigenvalue of the Sims (2002) augmented form of the linearized system, computes the derivative of its
    real and imaginary parts with respect to every free parameter. The result shows which parameters push eigenvalues
    toward or away from the unit circle, and so which can move the model in or out of Blanchard-Kahn stability.

    Eigenvalues are sorted by ascending modulus, so zeros come first and large or infinite eigenvalues last.

    Parameters
    ----------
    model : Model
        DSGE model.
    verbose : bool, optional
        Forwarded to the steady-state and linearization routines. Defaults to True.
    steady_state : dict, optional
        Steady state at which to evaluate the eigenvalues. Solved from ``model`` when None. Defaults to None.
    steady_state_kwargs : dict, optional
        Keyword arguments forwarded to ``model.steady_state``. Defaults to None.
    **parameter_updates
        Parameter values overriding the model defaults, for example ``beta=0.98``.

    Returns
    -------
    sensitivity : Dataset
        Dataset with two data variables. ``eigenvalues`` has shape ``(eigenvalue, component)`` with components
        ``real``, ``imaginary``, and ``modulus``. ``gradients`` has shape ``(eigenvalue, part, parameter)`` with parts
        ``real`` and ``imaginary``.

    Examples
    --------
    Find which parameters most move the eigenvalues that lie close to the unit circle:

    .. code-block:: python

        from gEconpy import model_from_gcn
        from gEconpy.data import get_example_gcn
        from gEconpy.model.statistics import eigenvalue_sensitivity

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        sensitivity = eigenvalue_sensitivity(model, verbose=False)

        modulus = sensitivity.eigenvalues.sel(component="modulus")
        near_unit_circle = sensitivity.eigenvalue.values[((modulus > 0.5) & (modulus < 2.0)).values]
        gradients = sensitivity.gradients.sel(eigenvalue=near_unit_circle, part="real")
    """
    if steady_state_kwargs is None:
        steady_state_kwargs = {}

    param_dict = model.parameters(**parameter_updates)
    if steady_state is None:
        steady_state = model.steady_state(**param_dict, verbose=verbose, **steady_state_kwargs)

    jacobians, ss_nodes, param_nodes, _eq_order, _var_order = model.symbolic_linearization(
        steady_state=steady_state, verbose=False
    )
    A_sym, B_sym, C_sym, D_sym = jacobians
    param_names = [p.name for p in param_nodes]

    # A, B, and C have columns in ``var_order``, so ``lead_var_idx`` (positions in the original variable order) is
    # translated to positions in the permuted column order.
    lead_var_idx = model.inv_var_order[model.lead_var_idx]
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
    model: "Model",
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
    """
    Check whether each row of ``samples`` yields a solvable DSGE model.

    Each row runs through the full solution pipeline: steady state, linearization, perturbation solve, Blanchard-Kahn
    check, and residual norms. The first stage that fails labels the row.

    Parameters
    ----------
    model : Model
        Compiled DSGE model. ``model.steady_state()`` must have been called at least once so the steady-state and
        linearization functions exist.
    samples : DataFrame
        Parameter draws, one per row. Column names must be a subset of the model parameter names. Parameters without
        a column keep the model's calibrated values.
    cores : int, optional
        Number of worker processes. Workers are forked where the platform allows it, so the model is shared without
        serialization. Defaults to 1.
    solver : str, optional
        Perturbation solver, one of ``"cycle_reduction"`` or ``"gensys"``. Backward-looking models always use
        ``"backward_direct"`` regardless of this setting. Defaults to ``"cycle_reduction"``.
    steady_state_kwargs : dict, optional
        Keyword arguments forwarded to ``model.steady_state()``. Defaults to None.
    linearize_kwargs : dict, optional
        Keyword arguments forwarded to ``model.linearize_model()``. Defaults to None.
    tol : float, optional
        Solver convergence tolerance. Defaults to 1e-8.
    max_iter : int, optional
        Maximum number of solver iterations. Defaults to 100.
    norm_tol : float, optional
        Threshold on the deterministic and stochastic residual norms. Defaults to 1e-8.
    progressbar : bool, optional
        Show a progress bar when ``cores`` is 1. Defaults to True.

    Returns
    -------
    results : DataFrame
        The input ``samples`` with three added columns. ``failure_step`` is None on success and otherwise names the
        first failing stage: ``"steady_state"``, ``"perturbation"``, ``"blanchard-kahn"``, ``"deterministic_norm"``,
        or ``"stochastic_norm"``. ``norm_deterministic`` and ``norm_stochastic`` hold the residual norms, or NaN when
        the pipeline failed before reaching them.

    Examples
    --------
    Check a hand-built grid of parameter values, then count how many rows failed at each stage:

    .. code-block:: python

        import pandas as pd

        from gEconpy import model_from_gcn, solvability_check
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        model.steady_state(verbose=False, progressbar=False)

        samples = pd.DataFrame({"alpha": [0.3, 0.35, 0.4], "beta": [0.98, 0.99, 0.995]})
        results = solvability_check(model, samples, progressbar=False)
        failures = results["failure_step"].value_counts(dropna=False)
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


def prior_solvability_check(
    model: "Model",
    n_samples: int,
    *,
    seed: int | np.random.Generator | None = None,
    param_subset: list[str] | None = None,
    method: str = "lhs",
    hdi_prob: float = 0.99,
    **kwargs,
) -> pd.DataFrame:
    """
    Draw parameters from the model priors and check whether each draw yields a solvable model.

    Draws a parameter DataFrame from the prior distributions declared in the GCN file, then delegates to
    :func:`solvability_check`.

    Parameters
    ----------
    model : Model
        DSGE model with at least one prior in ``param_priors``.
    n_samples : int
        Number of parameter draws.
    seed : int or Generator, optional
        Random seed. Defaults to None.
    param_subset : list of str, optional
        Parameters to sample. Every other parameter keeps the model's calibrated value. All parameters with priors
        are sampled when None. Defaults to None.
    method : str, optional
        Sampling strategy. ``"random"`` draws Monte Carlo samples from each prior. ``"lhs"``, ``"sobol"``,
        ``"halton"``, and ``"poisson_disk"`` draw uniform quasi-Monte Carlo samples over the prior HDI bounds.
        ``"sobol_ppf"`` and ``"halton_ppf"`` push quasi-Monte Carlo samples through the prior inverse CDF. Defaults
        to ``"lhs"``.
    hdi_prob : float, optional
        HDI probability defining the bounds of the uniform methods. Ignored by ``"random"`` and the ``"*_ppf"``
        methods. Defaults to 0.99.
    **kwargs
        Forwarded to :func:`solvability_check`.

    Returns
    -------
    results : DataFrame
        Sampled parameters with the columns described in :func:`solvability_check`.

    Examples
    --------
    Map out which prior draws fail to solve, and at which stage:

    .. code-block:: python

        from gEconpy import model_from_gcn, prior_solvability_check
        from gEconpy.data import get_example_gcn

        model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
        model.steady_state(verbose=False, progressbar=False)

        results = prior_solvability_check(model, n_samples=50, seed=1, progressbar=False)
        failures = results["failure_step"].value_counts(dropna=False)
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


def _collect_priors(
    model: "Model",
    param_subset: list[str] | None = None,
) -> dict:
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


def _run_serial(
    model: "Model",
    param_dicts: list[dict],
    shared_kwargs: dict,
    progressbar: bool,
) -> list[tuple]:
    if not progressbar:
        return [_check_one_draw(model, updates, **shared_kwargs) for updates in param_dicts]

    with Progress() as progress:
        task = progress.add_task("Checking solvability...", total=len(param_dicts))
        results = []
        for updates in param_dicts:
            results.append(_check_one_draw(model, updates, **shared_kwargs))
            progress.advance(task)
    return results


def _run_parallel(
    model: "Model",
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


def _check_one_draw(
    model: "Model",
    updates: dict,
    solver: str,
    steady_state_kwargs: dict,
    linearize_kwargs: dict,
    tol: float,
    max_iter: int,
    norm_tol: float,
) -> tuple[str | None, float, float]:
    """
    Run the full solvability pipeline for one parameter draw.

    Returns
    -------
    failure_step : str or None
        None on success, otherwise the name of the failing stage.
    deterministic_norm : float
        Deterministic residual norm, or NaN when the pipeline failed before reaching it.
    stochastic_norm : float
        Stochastic residual norm, or NaN when the pipeline failed before reaching it.
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
    """Dispatch to the named perturbation solver, returning ``(None, None)`` when it fails to find a solution."""
    effective_solver = "backward_direct" if backward_looking else solver

    if effective_solver == "cycle_reduction":
        T, R, _result, _log_norm = solve_policy_function_with_cycle_reduction(
            A, B, C, D, max_iter=max_iter, tol=tol, verbose=False
        )
        return T, R

    if effective_solver == "gensys":
        G_1, _const, impact, _f_mat, _f_wt, _y_wt, _gev, eu, _loose = solve_policy_function_with_gensys(A, B, C, D, tol)
        if not all(x == 1 for x in eu[:2]):
            return None, None
        return G_1[:n_variables, :n_variables], impact[:n_variables, :]

    if effective_solver == "backward_direct":
        return solve_policy_function_with_backward_direct(A, B, C, D)

    raise ValueError(f"Unknown solver {solver!r}. Pass 'cycle_reduction' or 'gensys'.")
