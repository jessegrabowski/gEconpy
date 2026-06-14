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
from gEconpy.model.compile import build_symbolic_jacobian
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
    the column scale is guarded as :math:`\mathrm{switch}(y_{ss} > 0,\; y_{ss},\; 1)`, so a non-positive steady
    state falls back to a level (un-logged) derivative rather than implying :math:`\log` of a non-positive value.

    Parameters
    ----------
    variables : list of TimeAwareSymbol
        Model variables, expressed at time t.
    equations : list of sp.Expr
        Model equations as sympy expressions.
    shocks : list of TimeAwareSymbol
        Exogenous shocks.
    cache : dict, optional
        Sympytensor cache mapping ``(name, assumptions)`` tuples to pytensor nodes. If provided, sympy-to-pytensor
        conversion reuses existing nodes. If None, a new cache is created.
    loglin_variables : list of TimeAwareSymbol, optional
        Variables to log-linearize. If None, all variables are log-linearized.
    order : int, default 1
        Order of approximation. Only ``order=1`` is currently supported.
    eq_order : ndarray of int, optional
        Permutation of equation indices placing equations in
        ``[static | lag-only | lead-only | both]`` order so A's and C's structural-zero
        row blocks become contiguous. Computed from the equations if not supplied.
    var_order : ndarray of int, optional
        Permutation of variable indices placing variables in
        ``[static | predetermined-only | mixed | forward-only]`` order so A's and C's
        structural-zero column blocks become contiguous. Computed from the equations if
        not supplied.

    Returns
    -------
    jacobians : list of TensorVariable
        Four pytensor matrix graph nodes ``[A, B, C, D]``. Rows are in ``eq_order`` and
        the variable axis (cols of A/B/C) is in ``var_order``; D's columns are shocks
        (no permutation).
    ss_input_nodes : list of TensorVariable
        Steady-state variable input nodes needed to evaluate the Jacobians. Parameter
        nodes are also embedded in the graph but must be discovered by the caller via
        ``explicit_graph_inputs``.
    eq_order_out : ndarray of int
        The equation permutation actually applied (same as ``eq_order`` if supplied).
    var_order_out : ndarray of int
        The variable permutation actually applied (same as ``var_order`` if supplied).
    """
    if order != 1:
        raise NotImplementedError("Only order = 1 linearization is currently implemented.")

    if cache is None:
        cache = {}

    n_vars = len(variables)
    lags, now, leads = make_all_variable_time_combinations(variables)

    if loglin_variables is None:
        loglin_set = set(range(n_vars))
    else:
        loglin_names = {v.base_name for v in loglin_variables}
        loglin_set = {i for i, v in enumerate(variables) if v.base_name in loglin_names}

    # Classify equations by which time-shifts they reference (S=static, L=lag-only, E=lead-only,
    # B=both) and variables by their incidence (s, p, m, f). A single pass over the equations
    # derives all four incidence arrays. These drive the [S|L|E|B] / [s|p|m|f] permutations that
    # make A's and C's structural-zero blocks contiguous for downstream solvers.
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

    if eq_order is None:
        eq_order_local = np.concatenate(
            [
                np.where(~eq_has_lag & ~eq_has_lead)[0],
                np.where(eq_has_lag & ~eq_has_lead)[0],
                np.where(~eq_has_lag & eq_has_lead)[0],
                np.where(eq_has_lag & eq_has_lead)[0],
            ]
        )
    else:
        eq_order_local = np.asarray(eq_order, dtype=int)
    if var_order is None:
        var_order_local = np.concatenate(
            [
                np.where(~var_has_lag & ~var_has_lead)[0],
                np.where(var_has_lag & ~var_has_lead)[0],
                np.where(var_has_lag & var_has_lead)[0],
                np.where(~var_has_lag & var_has_lead)[0],
            ]
        )
    else:
        var_order_local = np.asarray(var_order, dtype=int)

    # Order equations (rows) and variables (cols) up front so the matrices come out in
    # [eq_order, var_order] by construction -- no pytensor-side reshuffling.
    equations_perm = [equations[i] for i in eq_order_local]
    lags_perm = [lags[j] for j in var_order_local]
    now_perm = [now[j] for j in var_order_local]
    leads_perm = [leads[j] for j in var_order_local]

    # Bare derivatives at the steady state (vars -> ss, shocks -> 0). The shared ``cache``
    # makes common subexpressions and steady-state input nodes shared across A/B/C/D.
    A = build_symbolic_jacobian(equations_perm, lags_perm, cache, to_ss=True, shocks=shocks)
    B = build_symbolic_jacobian(equations_perm, now_perm, cache, to_ss=True, shocks=shocks)
    C = build_symbolic_jacobian(equations_perm, leads_perm, cache, to_ss=True, shocks=shocks)
    D = build_symbolic_jacobian(equations_perm, list(shocks), cache, to_ss=True, shocks=shocks)

    # Log-linear chain rule: scale column j (in var_order) by the steady-state factor of its
    # variable -- 1 for level variables (not in the loglin set, or declared negative), the
    # steady-state value for declared-positive variables, and a sign-guarded switch otherwise.
    column_scale = []
    for j in var_order_local:
        ss_node = as_tensor(variables[j].to_ss(), cache)
        assumptions = variables[j].assumptions0
        if j not in loglin_set or assumptions.get("negative", False):
            column_scale.append(pt.ones(()))
        elif assumptions.get("positive", False):
            column_scale.append(ss_node)
        else:
            column_scale.append(pt.switch(ss_node > 0, ss_node, pt.ones(())))
    scale = pt.stack(column_scale)

    A, B, C, D = rewrite_pregrad([A * scale, B * scale, C * scale, D])

    # Row order reflects the [S|L|E|B] equation permutation; column order reflects the
    # [s|p|m|f] variable permutation. Downstream solvers consume A/B/C/D and emit T/R in
    # the *permuted* variable order — the statespace boundary applies ``inv_var_order``
    # to map T/R back to user variable order, and ``Model.linearize_model`` applies
    # ``inv_eq_order`` / ``inv_var_order`` before returning matrices to the user.
    ss_pt = [as_tensor(v.to_ss(), cache) for v in variables]
    return [A, B, C, D], ss_pt, eq_order_local, var_order_local


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

    Returns a flag array where ``1`` means "do not log-linearize" and ``0`` means "log-linearize." Variables are
    excluded from log-linearization if they are explicitly listed in ``not_loglin_variables``, have steady-state
    values near zero, or (unless ``loglin_negative_ss`` is True) have negative steady-state values.

    Parameters
    ----------
    variables : list of TimeAwareSymbol
        Model variables.
    calibrated_params : list of sp.Symbol
        Calibrated parameters that also appear in the steady-state vector.
    steady_state : SymbolDictionary
        Steady-state values, keyed by variable name.
    log_linearize : bool, default True
        If False, returns all-ones (no variable is log-linearized).
    not_loglin_variables : list of str, optional
        Variable names the user explicitly excludes from log-linearization.
    loglin_negative_ss : bool, default False
        If True, variables with negative steady-state values are still log-linearized.
    verbose : bool, default True
        Log warnings about excluded variables.

    Returns
    -------
    flags : np.ndarray
        Array of length ``len(variables) + len(calibrated_params)``. Entry is ``1`` if the variable should not be
        log-linearized, ``0`` otherwise.
    """
    if not_loglin_variables is None:
        not_loglin_variables = []
    if not log_linearize:
        return np.ones(len(variables))

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
    B, C, D : np.ndarray
        Jacobian matrices from the linearized system.
    Q, P : np.ndarray
        Shock-response and transition sub-matrices of state variables.
    A_prime, R_prime, S_prime : np.ndarray
        Sub-matrices from ``statespace_to_gEcon_representation``.

    Returns
    -------
    norm_deterministic : float
        Frobenius norm of the deterministic residual
    norm_stochastic : float
        Frobenius norm of the stochastic residual
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
    Decompose the full state-space solution into gEcon's state/jumper partition.

    Splits the policy matrices ``T`` and ``R`` into state-variable and jumper-variable blocks according to which
    columns of ``T`` have at least one entry exceeding ``tol`` in absolute value.

    Parameters
    ----------
    A : np.ndarray
        Lead Jacobian matrix from the linearized system.
    T : np.ndarray
        Transition matrix (policy function for state evolution).
    R : np.ndarray
        Shock-response matrix.
    tol : float
        Threshold for identifying state variables.

    Returns
    -------
    P : np.ndarray
        State-to-state transition sub-matrix.
    Q : np.ndarray
        State shock-response sub-matrix.
    R : np.ndarray
        Jumper-to-state mapping sub-matrix.
    S : np.ndarray
        Jumper shock-response sub-matrix.
    A_prime : np.ndarray
        Lead Jacobian restricted to state-variable columns.
    R_prime : np.ndarray
        Transition matrix restricted to state-variable columns.
    S_prime : np.ndarray
        Shock-response matrix (all variables).
    """
    n_vars = T.shape[1]

    state_var_idx = np.where(np.abs(T[np.argmax(np.abs(T), axis=0), np.arange(n_vars)]) >= tol)[0]
    state_var_mask = np.isin(np.arange(n_vars), state_var_idx)

    PP = T.copy()
    PP[np.abs(PP) < tol] = 0
    QQ = R[:n_vars, :].copy()
    QQ[np.abs(QQ) < tol] = 0

    P = PP[state_var_mask, :][:, state_var_mask]
    Q = QQ[state_var_mask, :]
    R_out = PP[~state_var_mask, :][:, state_var_idx]
    S = QQ[~state_var_mask, :]

    A_prime = A[:, state_var_mask]
    R_prime = PP[:, state_var_mask]
    S_prime = QQ

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
    A, B, C, D : np.ndarray
        Jacobian matrices from the linearized system.
    T : np.ndarray
        Transition matrix.
    R : np.ndarray
        Shock-response matrix.
    tol : float, default 1e-8
        Tolerance for state-variable identification.
    """
    P, Q, _, _S, A_prime, R_prime, S_prime = statespace_to_gEcon_representation(A, T, R, tol)
    norm_det, norm_stoch = residual_norms(B, C, D, Q, P, A_prime, R_prime, S_prime)
    _log.info(f"Norm of deterministic part: {norm_det:0.9f}")
    _log.info(f"Norm of stochastic part:    {norm_stoch:0.9f}")


def compute_bk_eigenvalues(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute generalized eigenvalues of the linearized DSGE system for BK condition analysis.

    Builds the Sims (2002) augmented system and computes eigenvalues via ordered QZ decomposition.
    Eigenvalues are sorted by ascending modulus.

    Parameters
    ----------
    A, B, C, D : np.ndarray
        Jacobian matrices of the linearized DSGE system.
    tol : float, default 1e-8
        Tolerance for identifying forward-looking variables.

    Returns
    -------
    eigvals_real : np.ndarray
        Real parts of eigenvalues, sorted by modulus.
    eigvals_imag : np.ndarray
        Imaginary parts of eigenvalues, sorted by modulus.
    n_forward : int
        Number of forward-looking variables (columns of C with nonzero entries).
    """
    Gamma_0, Gamma_1, _, _, _ = _gensys_setup(A, B, C, D, tol)
    AA, BB, *_ = linalg.ordqz(-Gamma_0, Gamma_1, sort="ouc", output="complex")

    eigenvalues = np.diag(BB) / (np.diag(AA) + tol)
    idx = np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]

    n_forward = (np.abs(C).sum(axis=0) > tol).sum().astype(int)

    return np.real(eigenvalues), np.imag(eigenvalues), n_forward


def compute_bk_eigenvalues_pt(
    A: TensorVariable,
    B: TensorVariable,
    C: TensorVariable,
    _D: TensorVariable,
    lead_var_idx: np.ndarray,
) -> tuple[TensorVariable, TensorVariable]:
    """Compute symbolic eigenvalues of the linearized DSGE system for BK condition analysis.

    Builds the Sims (2002) augmented system symbolically and computes eigenvalues via
    ``real_eig``. The result is differentiable with respect to the input matrices.

    Parameters
    ----------
    A, B, C, _D : TensorVariable
        Jacobian matrices of the linearized DSGE system.
    lead_var_idx : np.ndarray of int
        Column indices of forward-looking variables (columns of C with nonzero entries).
        Must be known at graph-build time.

    Returns
    -------
    eigvals_real : TensorVariable
        Real parts of eigenvalues, sorted by modulus.
    eigvals_imag : TensorVariable
        Imaginary parts of eigenvalues, sorted by modulus.
    """
    lead_var_idx = np.asarray(lead_var_idx)
    n_vars = A.type.shape[0]
    if n_vars is None:
        raise ValueError("A must have a known static shape for symbolic BK eigenvalue computation.")

    I_n = pt.eye(n_vars)
    Z_n = pt.zeros((n_vars, n_vars))

    # Build augmented matrices (Sims 2002 form)
    Gamma_0 = pt.vertical_stack(
        pt.horizontal_stack(B, C),
        pt.horizontal_stack(-I_n, Z_n),
    )
    Gamma_1 = pt.vertical_stack(
        pt.horizontal_stack(A, Z_n),
        pt.horizontal_stack(Z_n, I_n),
    )

    # Row/column selection matching _gensys_setup:
    # all equation rows (0..n-1) + auxiliary rows for lead variables (n + lead_var_idx)
    eqs_and_leads_idx = np.concatenate([np.arange(n_vars), lead_var_idx + n_vars])
    Gamma_0_sel = Gamma_0[eqs_and_leads_idx, :][:, eqs_and_leads_idx]
    Gamma_1_sel = Gamma_1[eqs_and_leads_idx, :][:, eqs_and_leads_idx]

    # Gamma_0 may be singular (rank-deficient). Regularize with eps*I so that
    # infinite eigenvalues become O(1/eps) — still correctly counted as unstable —
    # while finite eigenvalues near |λ|=1 are perturbed by only O(eps).
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
    Check the Blanchard-Kahn condition for the linearized system.

    Computes the generalized eigenvalues of the system in the Sims (2002) [1]_ form. Per Blanchard and Kahn
    (1980) [2]_, the number of unstable eigenvalues (modulus > 1) must equal the number of forward-looking variables
    for a unique stable solution to exist.

    Parameters
    ----------
    A, B, C, D : np.ndarray
        Jacobian matrices of the linearized DSGE system, evaluated at steady state.
    tol : float, default 1e-8
        Tolerance below which numerical values are considered zero.
    verbose : bool, default True
        Whether to log the result.
    on_failure : ``'raise'`` or ``'ignore'``, default ``'ignore'``
        Action to take if the condition is not satisfied.
    return_value : ``'dataframe'``, ``'bool'``, or None, default ``'dataframe'``
        What to return: a DataFrame of eigenvalues, a boolean, or nothing.

    Returns
    -------
    result : pd.DataFrame, bool, or None
        Depends on ``return_value``:

        - ``'dataframe'``: DataFrame with columns ``Modulus``, ``Real``, ``Imaginary``.
        - ``'bool'``: True if the Blanchard-Kahn condition is satisfied.
        - ``None``: Nothing returned.

    References
    ----------
    .. [1] Sims, Christopher A. "Solving linear rational expectations models."
       *Computational Economics* 20.1-2 (2002): 1-20.
    .. [2] Blanchard, O.J. and Kahn, C.M. "The solution of linear difference models under
       rational expectations." *Econometrica* 48.5 (1980): 1305-1311.
    """
    if return_value not in ["dataframe", "bool", None]:
        raise ValueError(f'Unknown return type "{return_value}"')

    eigvals_real, eigvals_imag, n_forward = compute_bk_eigenvalues(A, B, C, D, tol)
    modulus = np.sqrt(eigvals_real**2 + eigvals_imag**2)
    n_unstable = (modulus > 1).sum()
    satisfied = n_forward == n_unstable

    message = (
        f"Model solution has {n_unstable} eigenvalues greater than one in modulus and {n_forward} "
        f"forward-looking variables.\nBlanchard-Kahn condition is{'' if satisfied else ' NOT'} satisfied."
    )

    if not satisfied:
        if n_unstable > n_forward:
            message += " No stable solution (more unstable eigenvalues than forward-looking variables)."
        else:
            message += " No unique solution (more forward-looking variables than unstable eigenvalues)."

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
    Symbolic Blanchard-Kahn condition check using differentiable eigenvalues.

    Parameters
    ----------
    A, B, C, D : TensorVariable
        Jacobian matrices of the linearized DSGE system.
    lead_var_idx : np.ndarray of int
        Column indices of forward-looking variables. Must be known at graph-build time.

    Returns
    -------
    bk_satisfied : TensorVariable (bool scalar)
        True if the Blanchard-Kahn condition IS satisfied.
    n_forward : TensorVariable (int scalar)
        Number of forward-looking variables.
    n_unstable : TensorVariable (int scalar)
        Number of eigenvalues with modulus greater than one.
    """
    lead_var_idx = np.asarray(lead_var_idx)
    n_forward = len(lead_var_idx)
    eigvals_real, eigvals_imag = compute_bk_eigenvalues_pt(A, B, C, D, lead_var_idx)

    # Detach the eigenvalues: the BK check is a step function (zero gradient), so this keeps
    # the only first-order-differentiable RealEig VJP out of the Hessian graph.
    eigvals_real = disconnected_grad(eigvals_real)
    eigvals_imag = disconnected_grad(eigvals_imag)

    modulus = pt.sqrt(eigvals_real**2 + eigvals_imag**2)
    n_unstable = (modulus > 1).sum()
    bk_satisfied = pt.eq(n_forward, n_unstable)

    return bk_satisfied, pt.constant(n_forward), n_unstable
