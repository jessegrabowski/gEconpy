import logging

from typing import Literal

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import sympy as sp

from pymc.pytensorf import rewrite_pregrad
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.graph.replace import graph_replace
from pytensor.tensor import TensorVariable
from scipy import linalg
from sympytensor import as_tensor

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.timing import make_all_variable_time_combinations
from gEconpy.pytensorf.sparse_jacobian import sparse_jacobian
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
) -> tuple[list[TensorVariable], list[TensorVariable]]:
    r"""
    Compute the log-linearized Jacobian matrices of a DSGE model using pytensor autodiff.

    Builds four Jacobian matrices ``A, B, C, D`` representing the first-order approximation of the model around its
    steady state:

    .. math::
        A \hat{y}_{t-1} + B \hat{y}_t + C \hat{y}_{t+1} + D \varepsilon_t = 0

    Log-linearization is implemented as a change-of-variables: for each variable to be log-linearized, the
    substitution :math:`y \to \exp(\tilde{y})` is applied before differentiation, then
    :math:`\tilde{y} \to \log(y)` is substituted back. The chain rule produces the classical T-matrix multiplication
    :math:`\partial F / \partial (\log y) = (\partial F / \partial y) \cdot y_{ss}`. Variables not in the loglin set
    use identity substitutions, yielding bare derivatives.

    A ``pt.switch`` guard on the steady-state value prevents ``log(negative)`` for variables whose steady states are
    non-positive. This is a numerical safety net: ``rewrite_pregrad`` does not simplify ``exp(log(x))`` when ``x < 0``.

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

    Returns
    -------
    jacobians : list of TensorVariable
        Four pytensor matrix graph nodes ``[A, B, C, D]``.
    ss_input_nodes : list of TensorVariable
        Steady-state variable input nodes needed to evaluate the Jacobians. Parameter nodes are also embedded in the
        graph but must be discovered by the caller via ``explicit_graph_inputs``.
    """
    if order != 1:
        raise NotImplementedError("Only order = 1 linearization is currently implemented.")

    if cache is None:
        cache = {}

    n_vars = len(variables)
    lags, now, leads = make_all_variable_time_combinations(variables)
    ss_vars = [v.to_ss() for v in variables]

    equations_pt = [as_tensor(eq, cache) for eq in equations]
    lags_pt = [as_tensor(v, cache) for v in lags]
    now_pt = [as_tensor(v, cache) for v in now]
    leads_pt = [as_tensor(v, cache) for v in leads]
    shocks_pt = [as_tensor(s, cache) for s in shocks]
    ss_pt = [as_tensor(v, cache) for v in ss_vars]

    if loglin_variables is None:
        loglin_set = set(range(n_vars))
    else:
        loglin_names = {v.base_name for v in loglin_variables}
        loglin_set = {i for i, v in enumerate(variables) if v.base_name in loglin_names}

    forward_replace = {}
    backward_replace = {}
    dummies_lags, dummies_now, dummies_leads = [], [], []

    for i, (lag, curr, lead, ss) in enumerate(zip(lags_pt, now_pt, leads_pt, ss_pt, strict=False)):
        name = variables[i].base_name
        lag_tilde = pt.dscalar(f"{name}__tm1__tilde")
        curr_tilde = pt.dscalar(f"{name}__t__tilde")
        lead_tilde = pt.dscalar(f"{name}__tp1__tilde")

        dummies_lags.append(lag_tilde)
        dummies_now.append(curr_tilde)
        dummies_leads.append(lead_tilde)

        if i in loglin_set:
            positive_ss = ss > 0
            for var, tilde in [(lag, lag_tilde), (curr, curr_tilde), (lead, lead_tilde)]:
                forward_replace[var] = pt.switch(positive_ss, pt.exp(tilde), tilde)
                backward_replace[tilde] = pt.switch(positive_ss, pt.log(var), var)
        else:
            for var, tilde in [(lag, lag_tilde), (curr, curr_tilde), (lead, lead_tilde)]:
                forward_replace[var] = tilde
                backward_replace[tilde] = var

    # Forward: y -> exp(y_tilde), differentiate w.r.t. y_tilde, then backward: y_tilde -> log(y)
    equations_transformed = graph_replace(equations_pt, forward_replace, strict=False)
    if not isinstance(equations_transformed, list):
        equations_transformed = [equations_transformed]
    equations_transformed = rewrite_pregrad(equations_transformed)

    A = sparse_jacobian(equations_transformed, dummies_lags, return_sparse=False)
    B = sparse_jacobian(equations_transformed, dummies_now, return_sparse=False)
    C = sparse_jacobian(equations_transformed, dummies_leads, return_sparse=False)
    D = sparse_jacobian(equations_transformed, shocks_pt, return_sparse=False) if shocks_pt else pt.zeros((n_vars, 1))

    A, B, C, D = rewrite_pregrad([graph_replace(m, backward_replace, strict=False) for m in [A, B, C, D]])

    # Evaluate at steady state: time-indexed variables -> ss values, shocks -> 0
    ss_replace = {}
    for lag, curr, lead, ss in zip(lags_pt, now_pt, leads_pt, ss_pt, strict=False):
        ss_replace[lag] = ss
        ss_replace[curr] = ss
        ss_replace[lead] = ss
    for shock in shocks_pt:
        ss_replace[shock] = pt.zeros(())

    A, B, C, D = [graph_replace(m, ss_replace, strict=False) for m in [A, B, C, D]]

    return [A, B, C, D], list(ss_pt)


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


def _compute_solution_eigenvalues(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, tol: float = 1e-8
) -> np.ndarray:
    """Compute sorted generalized eigenvalues of the linearized system via ordered QZ decomposition."""
    Gamma_0, Gamma_1, _, _, _ = _gensys_setup(A, B, C, D, tol)
    AA, BB, *_ = linalg.ordqz(-Gamma_0, Gamma_1, sort="ouc", output="complex")

    eigenvalues = np.diag(BB) / (np.diag(AA) + tol)

    eig = np.column_stack([np.abs(eigenvalues), np.real(eigenvalues), np.imag(eigenvalues)])
    return eig[np.argsort(eig[:, 0]), :]


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

    n_forward = (np.abs(C).sum(axis=0) > tol).sum().astype(int)
    eig = pd.DataFrame(_compute_solution_eigenvalues(A, B, C, D, tol), columns=["Modulus", "Real", "Imaginary"])
    n_unstable = (eig["Modulus"] > 1).sum()
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
        return eig
    return satisfied


class BlanchardKahnCondition(Op):
    """Pytensor Op wrapping the Blanchard-Kahn eigenvalue check for use in computation graphs."""

    def __init__(self, tol: float = 1e-8):
        self.tol = tol
        super().__init__()

    def make_node(self, A, B, C, D) -> Apply:
        inputs = list(map(pt.as_tensor, [A, B, C, D]))
        outputs = [
            pt.scalar("bk_flag", dtype=bool),
            pt.scalar("n_forward", dtype=int),
            pt.scalar("n_unstable", dtype=int),
        ]
        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        A, B, C, D = inputs
        n_forward = (np.abs(C).sum(axis=0) > _FLOAT_ZERO_TOL).sum().astype(int)
        eig = check_bk_condition(A, B, C, D, tol=self.tol, verbose=False, return_value="dataframe")
        n_unstable = (eig["Modulus"] > 1).sum()

        outputs[0][0] = np.array(n_forward != n_unstable)
        outputs[1][0] = np.array(n_forward)
        outputs[2][0] = np.array(n_unstable)


def check_bk_condition_pt(
    A: TensorVariable,
    B: TensorVariable,
    C: TensorVariable,
    D: TensorVariable,
    tol: float = 1e-8,
) -> tuple[TensorVariable, TensorVariable, TensorVariable]:
    r"""
    Pytensor wrapper for the Blanchard-Kahn condition check.

    Parameters
    ----------
    A, B, C, D : TensorVariable
        Jacobian matrices of the linearized DSGE system.
    tol : float, default 1e-8
        Tolerance below which numerical values are considered zero.

    Returns
    -------
    bk_flag : TensorVariable (bool scalar)
        True if the condition is **not** satisfied.
    n_forward : TensorVariable (int scalar)
        Number of forward-looking variables.
    n_unstable : TensorVariable (int scalar)
        Number of eigenvalues with modulus greater than one.

    References
    ----------
    .. [1] Sims, Christopher A. "Solving linear rational expectations models."
       *Computational Economics* 20.1-2 (2002): 1-20.
    .. [2] Blanchard, O.J. and Kahn, C.M. "The solution of linear difference models under
       rational expectations." *Econometrica* 48.5 (1980): 1305-1311.
    """
    return BlanchardKahnCondition(tol=tol)(A, B, C, D)
