import logging

from typing import Literal

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import sympy as sp

from pytensor.gradient import disconnected_grad
from pytensor.tensor import TensorVariable
from scipy import linalg
from sympytensor import as_tensor

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import build_symbolic_jacobians
from gEconpy.model.timing import make_all_variable_time_combinations
from gEconpy.pytensorf.compile import rewrite_pregrad
from gEconpy.pytensorf.real_eig import real_eig
from gEconpy.solvers.gensys import _gensys_setup
from gEconpy.utilities import get_name

_log = logging.getLogger(__name__)

_FLOAT_ZERO_TOL = 1e-8


def linearize_model(
    variables: list[TimeAwareSymbol],
    equations: list[sp.Expr],
    shocks: list[TimeAwareSymbol],
    cache: dict | None = None,
    loglin_variables: list[TimeAwareSymbol] | None = None,
    order: int = 1,
    eq_order: np.ndarray | None = None,
    var_order: np.ndarray | None = None,
) -> tuple[list[TensorVariable], list[TensorVariable], np.ndarray, np.ndarray]:
    r"""
    Compute the log-linearized Jacobian matrices of a DSGE model using pytensor autodiff.

    Builds four Jacobian matrices ``A, B, C, D`` representing the first-order approximation of the model around its
    steady state:

    .. math::
        A \hat{y}_{t-1} + B \hat{y}_t + C \hat{y}_{t+1} + D \varepsilon_t = 0

    Log-linearization applies the chain rule directly: :math:`\partial F / \partial (\log y) =
    (\partial F / \partial y) \cdot y_{ss}`. Each Jacobian is built by symbolic differentiation of the model
    equations, evaluated at the steady state, with the column for every log-linearized variable scaled by its
    steady-state value. Variables not in the loglin set keep their bare derivatives (scale factor one).

    For a log-linearized variable whose steady-state sign is not statically known from the GCN Assumptions block,
    the column scale is guarded as :math:`\mathrm{switch}(y_{ss} > 0,\; y_{ss},\; 1)`. A non-positive steady state
    then gets a level (un-logged) derivative, and :math:`\log` of a non-positive value never enters.

    Parameters
    ----------
    variables : list of TimeAwareSymbol
        Model variables, expressed at time t.
    equations : list of sp.Expr
        Model equations as sympy expressions.
    shocks : list of TimeAwareSymbol
        Exogenous shocks.
    cache : dict, optional
        Sympytensor cache mapping ``(name, assumptions)`` tuples to pytensor nodes, so sympy-to-pytensor conversion
        reuses existing nodes. A fresh cache is created when None. Defaults to None.
    loglin_variables : list of TimeAwareSymbol, optional
        Variables to log-linearize. Every variable is log-linearized when None. Defaults to None.
    order : int, optional
        Order of approximation. Only ``order=1`` is supported. Defaults to 1.
    eq_order : ndarray of int, optional
        Permutation of equation indices placing equations in ``[static | lag-only | lead-only | both]`` order, so the
        structural-zero row blocks of A and C are contiguous. Computed from the equations when None. Defaults to None.
    var_order : ndarray of int, optional
        Permutation of variable indices placing variables in ``[static | predetermined-only | mixed | forward-only]``
        order, so the structural-zero column blocks of A and C are contiguous. Computed from the equations when None.
        Defaults to None.

    Returns
    -------
    jacobians : list of TensorVariable
        Four pytensor matrix graph nodes ``[A, B, C, D]``. Rows follow ``eq_order`` and the variable axis (columns of
        A, B, and C) follows ``var_order``. The columns of D are shocks and carry no permutation.
    ss_input_nodes : list of TensorVariable
        Steady-state variable input nodes needed to evaluate the Jacobians. Parameter nodes are embedded in the graph
        and must be discovered by the caller with ``explicit_graph_inputs``.
    eq_order_out : ndarray of int
        The equation permutation applied, equal to ``eq_order`` when supplied.
    var_order_out : ndarray of int
        The variable permutation applied, equal to ``var_order`` when supplied.
    """
    if order != 1:
        raise NotImplementedError("Only order = 1 linearization is currently implemented.")

    if cache is None:
        cache = {}

    lags, now, leads = make_all_variable_time_combinations(variables)

    if loglin_variables is None:
        loglin_set = set(range(len(variables)))
    else:
        loglin_names = {v.base_name for v in loglin_variables}
        loglin_set = {i for i, v in enumerate(variables) if v.base_name in loglin_names}

    default_eq_order, default_var_order = _structural_orderings(variables, equations)
    eq_order_local = default_eq_order if eq_order is None else np.asarray(eq_order, dtype=int)
    var_order_local = default_var_order if var_order is None else np.asarray(var_order, dtype=int)

    # Permuting the equations and variables before differentiation yields matrices in [eq_order, var_order] by
    # construction, with no pytensor-side reshuffling.
    equations_perm = [equations[i] for i in eq_order_local]
    lags_perm = [lags[j] for j in var_order_local]
    now_perm = [now[j] for j in var_order_local]
    leads_perm = [leads[j] for j in var_order_local]

    # A, B, C, and D are derivatives of the same equations and share heavily, so one CSE pass across all four shrinks
    # the graph that pt.grad later replicates.
    A, B, C, D = build_symbolic_jacobians(
        [
            (equations_perm, lags_perm),
            (equations_perm, now_perm),
            (equations_perm, leads_perm),
            (equations_perm, list(shocks)),
        ],
        cache,
        to_ss=True,
        shocks=shocks,
    )

    scale = _log_linear_column_scale(variables, var_order_local, loglin_set, cache)
    A, B, C, D = rewrite_pregrad([A * scale, B * scale, C * scale, D])

    # Downstream solvers consume A, B, C, and D and emit T and R in the permuted variable order. The statespace
    # boundary and ``Model.linearize_model`` apply the inverse permutations before returning matrices to the user.
    ss_pt = [as_tensor(v.to_ss(), cache) for v in variables]
    return [A, B, C, D], ss_pt, eq_order_local, var_order_local


def _structural_orderings(variables: list[TimeAwareSymbol], equations: list[sp.Expr]) -> tuple[np.ndarray, np.ndarray]:
    """
    Order equations and variables so the structural-zero blocks of A and C are contiguous.

    Equations go into ``[static | lag-only | lead-only | both]`` order and variables into
    ``[static | predetermined-only | mixed | forward-only]`` order.
    """
    n_vars = len(variables)
    lag_syms = [v.set_t(-1) for v in variables]
    lead_syms = [v.set_t(1) for v in variables]

    eq_has_lag = np.zeros(len(equations), dtype=bool)
    eq_has_lead = np.zeros(len(equations), dtype=bool)
    var_has_lag = np.zeros(n_vars, dtype=bool)
    var_has_lead = np.zeros(n_vars, dtype=bool)
    for i, eq in enumerate(equations):
        atoms = eq.atoms(TimeAwareSymbol)
        for j in range(n_vars):
            if lag_syms[j] in atoms:
                eq_has_lag[i] = var_has_lag[j] = True
            if lead_syms[j] in atoms:
                eq_has_lead[i] = var_has_lead[j] = True

    eq_order = np.concatenate(
        [
            np.where(~eq_has_lag & ~eq_has_lead)[0],
            np.where(eq_has_lag & ~eq_has_lead)[0],
            np.where(~eq_has_lag & eq_has_lead)[0],
            np.where(eq_has_lag & eq_has_lead)[0],
        ]
    )
    var_order = np.concatenate(
        [
            np.where(~var_has_lag & ~var_has_lead)[0],
            np.where(var_has_lag & ~var_has_lead)[0],
            np.where(var_has_lag & var_has_lead)[0],
            np.where(~var_has_lag & var_has_lead)[0],
        ]
    )
    return eq_order, var_order


def _log_linear_column_scale(
    variables: list[TimeAwareSymbol],
    var_order: np.ndarray,
    loglin_set: set[int],
    cache: dict,
) -> TensorVariable:
    """
    Build the per-column chain-rule factor, in ``var_order``, that turns level derivatives into log-linear ones.

    The factor is 1 for level variables (outside ``loglin_set``, or declared negative), the steady-state value for
    declared-positive variables, and a sign-guarded switch otherwise.
    """
    column_scale = []
    for j in var_order:
        ss_node = as_tensor(variables[j].to_ss(), cache)
        assumptions = variables[j].assumptions0
        if j not in loglin_set or assumptions.get("negative", False):
            column_scale.append(pt.ones(()))
        elif assumptions.get("positive", False):
            column_scale.append(ss_node)
        else:
            column_scale.append(pt.switch(ss_node > 0, ss_node, pt.ones(())))
    return pt.stack(column_scale)


def make_not_loglin_flags(
    variables: list[TimeAwareSymbol],
    calibrated_params: list[sp.Symbol],
    steady_state: SymbolDictionary[str, float],
    log_linearize: bool = True,
    not_loglin_variables: list[str] | None = None,
    loglin_negative_ss: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Determine which variables should not be log-linearized.

    A variable is excluded from log-linearization when it is listed in ``not_loglin_variables``, when its steady-state
    value is near zero, or when its steady-state value is negative and ``loglin_negative_ss`` is False.

    Parameters
    ----------
    variables : list of TimeAwareSymbol
        Model variables.
    calibrated_params : list of sp.Symbol
        Calibrated parameters that also appear in the steady-state vector.
    steady_state : SymbolDictionary
        Steady-state values, keyed by variable name.
    log_linearize : bool, optional
        When False, no variable is log-linearized and the flags are all ones. Defaults to True.
    not_loglin_variables : list of str, optional
        Variable names the user excludes from log-linearization. Defaults to None.
    loglin_negative_ss : bool, optional
        Log-linearize variables with negative steady-state values. Defaults to False.
    verbose : bool, optional
        Log a warning naming each excluded variable. Defaults to True.

    Returns
    -------
    flags : ndarray
        Array of length ``len(variables) + len(calibrated_params)``. An entry is 1 when the variable should not be
        log-linearized and 0 otherwise.
    """
    if not_loglin_variables is None:
        not_loglin_variables = []
    if not log_linearize:
        return np.ones(len(variables) + len(calibrated_params))

    vars_and_calibrated = variables + calibrated_params
    var_names = [get_name(x, base_name=True) for x in vars_and_calibrated]
    unknown = set(not_loglin_variables) - set(var_names)

    if unknown:
        raise ValueError(
            f"The following variables were requested not to be log-linearized, but are unknown to the model: "
            f"{', '.join(unknown)}"
        )

    if verbose and not_loglin_variables:
        _log.warning(
            f"The following variables will not be log-linearized at the user's request: {not_loglin_variables}"
        )

    flags = np.array([name in not_loglin_variables for name in var_names], dtype=float)

    ss_values = np.array(list(steady_state.values()))
    ss_near_zero = np.abs(ss_values) < _FLOAT_ZERO_TOL
    ss_negative = ss_values < 0.0

    if np.any(ss_near_zero):
        zero_vars = [vars_and_calibrated[i] for i in np.flatnonzero(ss_near_zero)]
        if verbose:
            _log.warning(
                f"The following variables had steady-state values close to zero and will not be log-linearized:"
                f"{[get_name(x) for x in zero_vars]}"
            )
        flags[ss_near_zero] = 1

    if np.any(ss_negative) and not loglin_negative_ss:
        neg_vars = [vars_and_calibrated[i] for i in np.flatnonzero(ss_negative)]
        if verbose:
            _log.warning(
                f"The following variables had negative steady-state values and will not be log-linearized:"
                f"{[get_name(x) for x in neg_vars]}"
            )
        flags[ss_negative] = 1

    return flags


def residual_norms(
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    Q: np.ndarray,
    P: np.ndarray,
    A_prime: np.ndarray,
    R_prime: np.ndarray,
    S_prime: np.ndarray,
) -> tuple[float, float]:
    """
    Compute the norm of the deterministic and stochastic residuals of a solved perturbation system.

    Parameters
    ----------
    B, C, D : ndarray
        Jacobian matrices of the linearized system.
    Q, P : ndarray
        Shock-response and transition sub-matrices of the state variables.
    A_prime, R_prime, S_prime : ndarray
        Sub-matrices from :func:`statespace_to_gEcon_representation`.

    Returns
    -------
    norm_deterministic : float
        Frobenius norm of the deterministic residual.
    norm_stochastic : float
        Frobenius norm of the stochastic residual.
    """
    norm_deterministic = linalg.norm(A_prime + B @ R_prime + C @ R_prime @ P)
    norm_stochastic = linalg.norm(B @ S_prime + C @ R_prime @ Q + D)
    return norm_deterministic, norm_stochastic


def statespace_to_gEcon_representation(
    A: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose the state space solution into the state and jumper partition used by gEcon.

    A variable is a state when its column of ``T`` has at least one entry of at least ``tol`` in absolute value.
    Entries of ``T`` and ``R`` below ``tol`` in absolute value are set to zero before partitioning.

    Parameters
    ----------
    A : ndarray
        Jacobian of the system with respect to variables at t-1.
    T : ndarray
        Transition matrix.
    R : ndarray
        Selection matrix.
    tol : float
        Threshold for identifying state variables and for zeroing small entries.

    Returns
    -------
    P : ndarray
        State-to-state transition sub-matrix.
    Q : ndarray
        State shock-response sub-matrix.
    R : ndarray
        Jumper-to-state mapping sub-matrix.
    S : ndarray
        Jumper shock-response sub-matrix.
    A_prime : ndarray
        ``A`` restricted to state-variable columns.
    R_prime : ndarray
        Transition matrix restricted to state-variable columns.
    S_prime : ndarray
        Selection matrix for all variables.
    """
    n_vars = T.shape[1]

    column_max = np.abs(T).max(axis=0)
    state_var_mask = column_max >= tol

    T_thresholded = T.copy()
    T_thresholded[np.abs(T_thresholded) < tol] = 0
    R_thresholded = R[:n_vars, :].copy()
    R_thresholded[np.abs(R_thresholded) < tol] = 0

    P = T_thresholded[state_var_mask, :][:, state_var_mask]
    Q = R_thresholded[state_var_mask, :]
    R_out = T_thresholded[~state_var_mask, :][:, state_var_mask]
    S = R_thresholded[~state_var_mask, :]

    A_prime = A[:, state_var_mask]
    R_prime = T_thresholded[:, state_var_mask]
    S_prime = R_thresholded

    return P, Q, R_out, S, A_prime, R_prime, S_prime


def check_perturbation_solution(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    tol: float = 1e-8,
) -> None:
    """
    Log the residual norms of a solved perturbation system.

    Parameters
    ----------
    A, B, C, D : ndarray
        Jacobian matrices of the linearized system.
    T : ndarray
        Transition matrix.
    R : ndarray
        Selection matrix.
    tol : float, optional
        Threshold for identifying state variables. Defaults to 1e-8.
    """
    P, Q, _, _S, A_prime, R_prime, S_prime = statespace_to_gEcon_representation(A, T, R, tol)
    norm_det, norm_stoch = residual_norms(B, C, D, Q, P, A_prime, R_prime, S_prime)
    _log.info(f"Norm of deterministic part: {norm_det:0.9f}")
    _log.info(f"Norm of stochastic part:    {norm_stoch:0.9f}")


def compute_bk_eigenvalues(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute the generalized eigenvalues of the linearized DSGE system for the Blanchard-Kahn check.

    Builds the Sims (2002) augmented system and computes its eigenvalues by ordered QZ decomposition. Eigenvalues are
    sorted by ascending modulus.

    Parameters
    ----------
    A, B, C, D : ndarray
        Jacobian matrices of the linearized DSGE system.
    tol : float, optional
        Threshold on the absolute column sums of ``C`` below which a lead column counts as empty, and on the
        pencil diagonal below which an eigenvalue counts as infinite. Defaults to 1e-8.

    Returns
    -------
    eigvals_real : ndarray
        Real parts of the eigenvalues, sorted by modulus.
    eigvals_imag : ndarray
        Imaginary parts of the eigenvalues, sorted by modulus.
    n_lead_columns : int
        Number of columns of ``C`` whose absolute column sum exceeds ``tol``, the variables entering at ``t+1``
        at these parameter values. It is smaller than :attr:`~gEconpy.model.model.Model.n_forward` when every
        coefficient on some lead variable vanishes at the current parameterization.
    """
    G0, Gamma_1, _, _, _, lead_var_idx = _gensys_setup(A, B, C, D, tol)
    AA, BB, *_ = linalg.ordqz(-G0, Gamma_1, sort="ouc", output="complex")

    # A zero on the diagonal of AA is an infinite generalized eigenvalue, not a large one. Nudging the
    # denominator by tol instead would bias every eigenvalue, and would amplify rather than damp the ones where
    # the diagonal sits near -tol.
    alpha, beta = np.diag(AA), np.diag(BB)
    finite = np.abs(alpha) > tol
    eigenvalues = np.full(alpha.shape, np.inf, dtype=complex)
    eigenvalues[finite] = beta[finite] / alpha[finite]

    eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))]

    return np.real(eigenvalues), np.imag(eigenvalues), int(lead_var_idx.size)


def compute_bk_eigenvalues_pt(
    A: TensorVariable,
    B: TensorVariable,
    C: TensorVariable,
    _D: TensorVariable,
    lead_var_idx: np.ndarray,
) -> tuple[TensorVariable, TensorVariable]:
    """
    Compute the eigenvalues of the linearized DSGE system symbolically for the Blanchard-Kahn check.

    Builds the Sims (2002) augmented system and computes its eigenvalues with
    :func:`~gEconpy.pytensorf.real_eig.real_eig`, so the result is differentiable with respect to the input matrices.

    Parameters
    ----------
    A, B, C, _D : TensorVariable
        Jacobian matrices of the linearized DSGE system. ``_D`` is unused.
    lead_var_idx : ndarray of int
        Column indices of the forward-looking variables. Must be known at graph-build time.

    Returns
    -------
    eigvals_real : TensorVariable
        Real parts of the eigenvalues, sorted by modulus.
    eigvals_imag : TensorVariable
        Imaginary parts of the eigenvalues, sorted by modulus.
    """
    lead_var_idx = np.asarray(lead_var_idx)
    n_vars = A.type.shape[0]
    if n_vars is None:
        raise ValueError("A must have a known static shape for symbolic BK eigenvalue computation.")

    I_n = pt.eye(n_vars)
    Z_n = pt.zeros((n_vars, n_vars))

    Gamma_0 = pt.vertical_stack(
        pt.horizontal_stack(B, C),
        pt.horizontal_stack(-I_n, Z_n),
    )
    Gamma_1 = pt.vertical_stack(
        pt.horizontal_stack(A, Z_n),
        pt.horizontal_stack(Z_n, I_n),
    )

    # Same row and column selection as _gensys_setup: every equation row plus one auxiliary row per lead variable.
    eqs_and_leads_idx = np.concatenate([np.arange(n_vars), lead_var_idx + n_vars])
    Gamma_0_sel = Gamma_0[eqs_and_leads_idx, :][:, eqs_and_leads_idx]
    Gamma_1_sel = Gamma_1[eqs_and_leads_idx, :][:, eqs_and_leads_idx]

    # Gamma_0 may be singular. Regularizing with eps*I turns infinite eigenvalues into O(1/eps) ones, which are still
    # counted as unstable, while finite eigenvalues near the unit circle move by only O(eps).
    n_sel = len(eqs_and_leads_idx)
    G0_reg = -Gamma_0_sel + pt.eye(n_sel) * _FLOAT_ZERO_TOL
    M = pt.linalg.solve(G0_reg, Gamma_1_sel)
    return real_eig(M)


def check_bk_condition(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    tol: float = 1e-8,
    verbose: bool = True,
    on_failure: Literal["raise", "ignore"] = "ignore",
    return_value: Literal["dataframe", "bool", None] = "dataframe",
) -> bool | pd.DataFrame | None:
    r"""
    Check the Blanchard-Kahn condition of the linearized system.

    Computes the generalized eigenvalues of the system in the Sims (2002) [1]_ form. Per Blanchard and Kahn
    (1980) [2]_, a unique stable solution exists when the number of unstable eigenvalues (modulus greater than one)
    equals the number of forward-looking variables.

    Parameters
    ----------
    A, B, C, D : ndarray
        Jacobian matrices of the linearized DSGE system, evaluated at the steady state.
    tol : float, optional
        Threshold below which numerical values count as zero. Defaults to 1e-8.
    verbose : bool, optional
        Log the result. Defaults to True.
    on_failure : str, optional
        One of ``'raise'`` or ``'ignore'``. Action to take when the condition is not satisfied. Defaults to
        ``'ignore'``.
    return_value : str or None, optional
        One of ``'dataframe'``, ``'bool'``, or None. Selects what to return. Defaults to ``'dataframe'``.

    Returns
    -------
    result : DataFrame, bool, or None
        With ``'dataframe'``, a DataFrame with columns ``Modulus``, ``Real``, and ``Imaginary``. With ``'bool'``, True
        when the Blanchard-Kahn condition is satisfied. With None, nothing.

    References
    ----------
    .. [1] Sims, Christopher A. "Solving linear rational expectations models."
       *Computational Economics* 20.1-2 (2002): 1-20.
    .. [2] Blanchard, O.J. and Kahn, C.M. "The solution of linear difference models under
       rational expectations." *Econometrica* 48.5 (1980): 1305-1311.
    """
    if return_value not in ["dataframe", "bool", None]:
        raise ValueError(f'Unknown return_value "{return_value}". Pass "dataframe", "bool", or None.')

    eigvals_real, eigvals_imag, n_lead_columns = compute_bk_eigenvalues(A, B, C, D, tol)
    modulus = np.sqrt(eigvals_real**2 + eigvals_imag**2)
    n_unstable = int((modulus > 1).sum())
    satisfied = bool(n_lead_columns == n_unstable)

    counts = (
        f"{n_unstable} eigenvalues outside the unit circle against {n_lead_columns} forward-looking variables "
        f"(columns of the lead Jacobian that are nonzero at these parameters)"
    )

    if satisfied:
        message = f"Blanchard-Kahn condition satisfied: {counts}."
    else:
        consequence = "no stable solution" if n_unstable > n_lead_columns else "no unique solution"
        message = f"Blanchard-Kahn condition NOT satisfied: {counts}. The model has {consequence}."

    if not satisfied and on_failure == "raise":
        raise ValueError(message)

    if verbose:
        _log.info(message)

    if return_value is None:
        return None
    if return_value == "dataframe":
        return pd.DataFrame({"Modulus": modulus, "Real": eigvals_real, "Imaginary": eigvals_imag})
    return satisfied


def check_bk_condition_pt(
    A: TensorVariable,
    B: TensorVariable,
    C: TensorVariable,
    D: TensorVariable,
    lead_var_idx: np.ndarray,
) -> tuple[TensorVariable, TensorVariable, TensorVariable]:
    r"""
    Check the Blanchard-Kahn condition symbolically.

    Parameters
    ----------
    A, B, C, D : TensorVariable
        Jacobian matrices of the linearized DSGE system.
    lead_var_idx : ndarray of int
        Column indices of the forward-looking variables. Must be known at graph-build time.

    Returns
    -------
    bk_satisfied : TensorVariable
        Boolean scalar, True when the Blanchard-Kahn condition is satisfied.
    n_forward : TensorVariable
        Integer scalar, the number of forward-looking variables.
    n_unstable : TensorVariable
        Integer scalar, the number of eigenvalues with modulus greater than one.
    """
    lead_var_idx = np.asarray(lead_var_idx)
    n_forward = len(lead_var_idx)
    eigvals_real, eigvals_imag = compute_bk_eigenvalues_pt(A, B, C, D, lead_var_idx)

    # The BK check is a step function with zero gradient, so detaching the eigenvalues keeps the RealEig VJP, which
    # is only first-order differentiable, out of the Hessian graph.
    eigvals_real = disconnected_grad(eigvals_real)
    eigvals_imag = disconnected_grad(eigvals_imag)

    modulus = pt.sqrt(eigvals_real**2 + eigvals_imag**2)
    n_unstable = (modulus > 1).sum()
    bk_satisfied = pt.eq(n_forward, n_unstable)

    return bk_satisfied, pt.constant(n_forward), n_unstable
