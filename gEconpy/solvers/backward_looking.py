import numpy as np
import pytensor.tensor as pt

from pytensor.tensor.variable import TensorVariable
from scipy import linalg


def solve_policy_function_with_backward_direct(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,  # noqa: ARG001
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve for the policy function of a backward-looking model by direct linear solves.

    Parameters
    ----------
    A : ndarray
        Jacobian of the system with respect to variables at t-1.
    B : ndarray
        Jacobian of the system with respect to variables at t.
    C : ndarray
        Jacobian of the system with respect to variables at t+1. Zero in a backward-looking model, and accepted only
        so the signature matches the forward-looking solvers.
    D : ndarray
        Jacobian of the system with respect to exogenous shocks.

    Returns
    -------
    T : ndarray
        Transition matrix, giving the effect of variable values at t-1 on their values at t.
    R : ndarray
        Selection matrix, giving the effect of exogenous shocks on variable values.
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
    Build a symbolic graph that solves for the policy function of a backward-looking model.

    Parameters
    ----------
    A : TensorVariable
        Jacobian of the system with respect to variables at t-1.
    B : TensorVariable
        Jacobian of the system with respect to variables at t.
    C : TensorVariable
        Jacobian of the system with respect to variables at t+1. Zero in a backward-looking model, and accepted only
        so the signature matches the forward-looking solvers.
    D : TensorVariable
        Jacobian of the system with respect to exogenous shocks.

    Returns
    -------
    T : TensorVariable
        Transition matrix, giving the effect of variable values at t-1 on their values at t.
    R : TensorVariable
        Selection matrix, giving the effect of exogenous shocks on variable values.
    """
    T = solve_backward_policy_pt(A, B)
    R = solve_backward_shock_matrix_pt(B, D)

    return T, R


def solve_backward_policy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve for the transition matrix :math:`T = -B^{-1} A` of a backward-looking model.

    Parameters
    ----------
    A : ndarray
        Jacobian of the system with respect to variables at t-1.
    B : ndarray
        Jacobian of the system with respect to variables at t.

    Returns
    -------
    T : ndarray
        Transition matrix, giving the effect of variable values at t-1 on their values at t.
    """
    return linalg.solve(-B, A)


def solve_backward_policy_pt(A: TensorVariable, B: TensorVariable) -> TensorVariable:
    """
    Build a symbolic graph for the transition matrix :math:`T = -B^{-1} A` of a backward-looking model.

    Parameters
    ----------
    A : TensorVariable
        Jacobian of the system with respect to variables at t-1.
    B : TensorVariable
        Jacobian of the system with respect to variables at t.

    Returns
    -------
    T : TensorVariable
        Transition matrix, giving the effect of variable values at t-1 on their values at t.
    """
    return pt.linalg.solve(-B, A)


def solve_backward_shock_matrix(B: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Solve for the selection matrix of a backward-looking model.

    In the general case the selection matrix is

    .. math::

        R = -(C T + B)^{-1} D

    where :math:`C` is the Jacobian with respect to variables at t+1 and :math:`T` the transition matrix. In a
    backward-looking model :math:`C` is zero, so this reduces to

    .. math::

        R = -B^{-1} D

    Parameters
    ----------
    B : ndarray
        Jacobian of the system with respect to variables at t.
    D : ndarray
        Jacobian of the system with respect to exogenous shocks.

    Returns
    -------
    R : ndarray
        Selection matrix, giving the effect of exogenous shocks on variable values.
    """
    return -np.linalg.solve(B, D)


def solve_backward_shock_matrix_pt(B: TensorVariable, D: TensorVariable) -> TensorVariable:
    """
    Build a symbolic graph for the selection matrix :math:`R = -B^{-1} D` of a backward-looking model.

    See :func:`~gEconpy.solvers.backward_looking.solve_backward_shock_matrix` for the derivation.

    Parameters
    ----------
    B : TensorVariable
        Jacobian of the system with respect to variables at t.
    D : TensorVariable
        Jacobian of the system with respect to exogenous shocks.

    Returns
    -------
    R : TensorVariable
        Selection matrix, giving the effect of exogenous shocks on variable values.
    """
    return -pt.linalg.solve(B, D)
