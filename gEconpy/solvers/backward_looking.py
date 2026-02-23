import numpy as np
import pytensor.tensor as pt

from pytensor.tensor.variable import TensorVariable
from scipy import linalg


def solve_backward_policy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solves for the policy function in a backward-looking model.

    Parameters
    ----------
    A : np.ndarray
        Jacobian matrix with respect to variables at t-1.
    B : np.ndarray
        Jacobian matrix with respect to variables at t.

    Returns
    -------
    T : np.ndarray
        The policy function matrix that maps current state variables to control variables.
    """
    return linalg.solve(-B, A)


def solve_backward_policy_pt(A: TensorVariable, B: TensorVariable) -> TensorVariable:
    """
    Solves for the policy function in a backward-looking model using PyTensor.

    Parameters
    ----------
    A : TensorVariable
        Jacobian matrix with respect to variables at t-1.
    B : TensorVariable
        Jacobian matrix with respect to variables at t.

    Returns
    -------
    T : TensorVariable
        The policy function matrix that maps current state variables to control variables.
    """
    return pt.linalg.solve(-B, A)


def solve_backward_shock_matrix(B: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Solves for the shock matrix in a backward-looking model.

    The equation for the shock matrix R in the general case is:

    .. math ::

        R = -(C T + B)^{-1} D

    Where :math:`C` is the Jacobian matrix with respect to variables at t+1, :math:`T` is the policy function matrix,
    :math:`B` is the Jacobian matrix with respect to variables at t, and :math:`D` is the Jacobian matrix with respect
    to exogenous shocks.

    Since the :math:`C` matrix is zero in backward-looking models, this equation simplifies to:

    .. math ::

        R = -B^{-1} D

    Parameters
    ----------
    B : np.ndarray
        Jacobian matrix with respect to variables at t.
    D : np.ndarray
        Jacobian matrix with respect exogenous shock variables.

    Returns
    -------
    R : np.ndarray
        The shock matrix that maps current state variables to future state variables.
    """
    return -np.linalg.solve(B, D.astype(B.dtype))


def solve_backward_shock_matrix_pt(B: TensorVariable, D: TensorVariable) -> TensorVariable:
    """
    Solves for the shock matrix in a backward-looking model using PyTensor.

    For details, see :func:`solve_backward_shock_matrix`.

    Parameters
    ----------
    B : TensorVariable
        Jacobian matrix with respect to variables at t.
    D : TensorVariable
        Jacobian matrix with respect exogenous shock variables.

    Returns
    -------
    R : TensorVariable
        The shock matrix that maps current state variables to future state variables.
    """
    return -pt.linalg.solve(B, D)


def solve_policy_function_with_backward_direct(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,  # noqa: ARG001
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves for the policy function in a backward-looking model using a direct method.

    Parameters
    ----------
    A : np.ndarray
        Jacobian matrix with respect to variables at t-1.
    B : np.ndarray
        Jacobian matrix with respect to variables at t.
    C : np.ndarray
        Jacobian matrix with respect to variables at t+1. Assumed to be zero in backward-looking models; included
        for API consistency with forward-looking models.
    D : np.ndarray
        Jacobian matrix with respect to variables exogenous shocks.

    Returns
    -------
    T : np.ndarray
        The policy function matrix that maps current state variables to control variables.
    R : np.ndarray
        The transition matrix that maps current state variables to future state variables.
    """
    T = solve_backward_policy(A, B)
    R = solve_backward_shock_matrix(B, D)

    return T, R


def solve_policy_function_with_backward_direct_pt(
    A: TensorVariable,
    B: TensorVariable,
    C: TensorVariable,  # noqa: ARG001
    D: TensorVariable,
) -> tuple[TensorVariable, TensorVariable]:
    """
    Solves for the policy function in a backward-looking model using a direct method with PyTensor.

    For details, see :func:`solve_policy_function_with_backward_direct`.

    Parameters
    ----------
    A : TensorVariable
        Jacobian matrix with respect to variables at t-1.
    B : TensorVariable
        Jacobian matrix with respect to variables at t.
    C : TensorVariable
        Jacobian matrix with respect to variables at t+1. Assumed to be zero in backward-looking models; included
        for API consistency with forward-looking models.
    D : TensorVariable
        Jacobian matrix with respect to exogenous shocks.

    Returns
    -------
    T : TensorVariable
        The policy function matrix that maps current state variables to control variables.
    R : TensorVariable
        The transition matrix that maps current state variables to future state variables.
    """
    T = solve_backward_policy_pt(A, B)
    R = solve_backward_shock_matrix_pt(B, D)

    return T, R
