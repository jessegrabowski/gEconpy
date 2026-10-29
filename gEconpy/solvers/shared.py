from collections.abc import Callable

import numpy as np
import pytensor.tensor as pt

from pytensor.tensor import TensorVariable
from pytensor.tensor.symbolic import TensorSymbolicOp

from gEconpy.solvers.stein import AdjointMethod, solve_policy_stein


def stabilize(x: TensorVariable, jitter: float = 1e-16) -> TensorVariable:
    """
    Add a small constant to the diagonal of a square matrix to keep linear solves well conditioned.

    Parameters
    ----------
    x : TensorVariable
        Square matrix to stabilize.
    jitter : float, optional
        Amount added to each diagonal entry. Defaults to 1e-16.

    Returns
    -------
    x_stable : TensorVariable
        Copy of ``x`` with ``jitter`` added to its diagonal.
    """
    n = x.shape[0]
    idx = pt.arange(n)
    return x[idx, idx].inc(jitter)


def o1_policy_function_adjoints(
    A: TensorVariable,  # noqa: ARG001
    B: TensorVariable,
    C: TensorVariable,
    T: TensorVariable,
    T_bar: TensorVariable,
    method: AdjointMethod = "doubling",
) -> list[TensorVariable]:
    r"""
    Compute the adjoints of the matrix quadratic equation that defines a first-order DSGE policy function.

    The policy function :math:`T` solves

    .. math::

        A + BT + CTT = 0

    Given the cotangent :math:`\bar{T}` of a scalar loss with respect to :math:`T`, the adjoints of :math:`A`,
    :math:`B`, and :math:`C` follow from the Lagrange multiplier :math:`S` of the constraint, which solves the
    generalized Stein equation

    .. math::

        (B + CT)^T S + C^T S T^T = -\bar{T}

    Parameters
    ----------
    A : TensorVariable
        Jacobian of the system with respect to variables at t-1, evaluated at the steady state.
    B : TensorVariable
        Jacobian of the system with respect to variables at t, evaluated at the steady state.
    C : TensorVariable
        Jacobian of the system with respect to variables at t+1, evaluated at the steady state.
    T : TensorVariable
        Solved policy function matrix, such that :math:`X_t = T X_{t-1}`.
    T_bar : TensorVariable
        Cotangent of a scalar loss with respect to ``T``.
    method : {"doubling", "qz", "sylvester", "direct"}, optional
        Algorithm for the Stein equation. See :func:`~gEconpy.solvers.stein.solve_policy_stein` for the trade-offs.
        Defaults to ``"doubling"``.

    Returns
    -------
    adjoints : list of TensorVariable
        Adjoints of ``A``, ``B`` and ``C``, in that order.
    """
    S = solve_policy_stein((B + C @ T).T, C, T, -T_bar, method=method)

    A_bar = S
    B_bar = S @ T.T
    C_bar = S @ T.T @ T.T

    return [A_bar, B_bar, C_bar]


def pt_compute_selection_matrix(
    B: TensorVariable,
    C: TensorVariable,
    D: TensorVariable,
    T: TensorVariable,
) -> TensorVariable:
    """
    Compute the shock selection matrix :math:`R = -(CT + B)^{-1} D`.

    Parameters
    ----------
    B : TensorVariable
        Jacobian of the system with respect to variables at t, evaluated at the steady state.
    C : TensorVariable
        Jacobian of the system with respect to variables at t+1, evaluated at the steady state.
    D : TensorVariable
        Jacobian of the system with respect to exogenous shocks, evaluated at the steady state.
    T : TensorVariable
        Solved policy function matrix.

    Returns
    -------
    R : TensorVariable
        Selection matrix giving the effect of exogenous shocks on the variables.
    """
    return -pt.linalg.solve(C @ T + B, D, assume_a="gen", check_finite=False)


def solve_with_static_elimination(
    solver: Callable[..., tuple[TensorVariable, ...]],
    A: TensorVariable,
    B: TensorVariable,
    C: TensorVariable,
    D: TensorVariable,
    static_mask: np.ndarray,
) -> tuple[TensorVariable, ...]:
    r"""
    Solve a linearized DSGE system after eliminating its static variables.

    A static variable appears only at time t, so its columns of ``A`` and ``C`` are zero. Let :math:`Q = [Q_1, Q_2]`
    be an orthogonal basis whose first block spans the static columns :math:`B_s` of ``B``. The rows
    :math:`Q_2^T - (Q_2^T B_s)(Q_1^T B_s)^{-1} Q_1^T` annihilate :math:`B_s`, so they leave a quadratic matrix equation
    in the dynamic variables alone, which ``solver`` solves. The rows :math:`Q_1^T` then give the static rows of
    :math:`T` and :math:`R` by a solve against :math:`Q_1^T B_s`.

    Gradients with respect to ``B``, ``D``, and the dynamic columns of ``A`` and ``C`` are those of the full system.
    The reduced solve never reads the static columns of ``A`` and ``C``, so their gradients are zero. When
    :math:`B_s` is numerically singular, every entry of :math:`T` and :math:`R` is NaN.

    Parameters
    ----------
    solver : callable
        First-order solver called as ``solver(A, B, C, D)`` on the reduced system, returning ``T``, ``R``, and any
        further outputs, which are passed through unchanged.
    A : TensorVariable
        Jacobian of the system with respect to variables at t-1, evaluated at the steady state.
    B : TensorVariable
        Jacobian of the system with respect to variables at t, evaluated at the steady state.
    C : TensorVariable
        Jacobian of the system with respect to variables at t+1, evaluated at the steady state.
    D : TensorVariable
        Jacobian of the system with respect to exogenous shocks, evaluated at the steady state.
    static_mask : ndarray of bool
        Whether each column of ``A``, ``B`` and ``C`` belongs to a static variable. A marked column must be a
        structural zero in both ``A`` and ``C``.

    Returns
    -------
    T : TensorVariable
        Transition matrix of the full system.
    R : TensorVariable
        Selection matrix of the full system.
    *extras : TensorVariable
        Remaining outputs of ``solver``.
    """
    static_mask = np.asarray(static_mask, dtype=bool)
    static_idx = np.flatnonzero(static_mask)
    dynamic_idx = np.flatnonzero(~static_mask)
    n_static = static_idx.size
    if n_static == 0:
        return solver(A, B, C, D)
    if dynamic_idx.size == 0:
        raise ValueError("Every variable is static, so there is no dynamic system to solve.")

    # The annihilating rows remove B_s for any invertible Q, so T and R do not depend on Q and its gradient is zero. The
    # dependence on B_s that the gradient needs runs through the oblique correction, which is zero in value.
    Q = _constant_orthogonal_basis(B[:, static_idx])

    # Rotating and reducing the column-stacked [B_s | A_d | B_d | C_d | D] takes one product each instead of one per
    # block, which keeps the forward graph and its gradient small.
    n_dynamic = dynamic_idx.size
    stacked = pt.concatenate([B[:, static_idx], A[:, dynamic_idx], B[:, dynamic_idx], C[:, dynamic_idx], D], axis=1)
    rotated = Q.T @ stacked
    top, bottom = rotated[:n_static], rotated[n_static:]

    B_s_top = top[:, :n_static]
    correction = pt.linalg.solve(B_s_top.T, bottom[:, :n_static].T, assume_a="gen", check_finite=False).T
    reduced = bottom[:, n_static:] - correction @ top[:, n_static:]

    block_widths = pt.stack([n_dynamic, n_dynamic, n_dynamic, D.shape[1]])
    T_dynamic, R_dynamic, *extras = solver(*pt.split(reduced, block_widths, n_splits=4, axis=1))

    A_top, B_top, C_top, D_top = pt.split(top[:, n_static:], block_widths, n_splits=4, axis=1)
    T_static = -pt.linalg.solve(
        B_s_top, A_top + B_top @ T_dynamic + C_top @ T_dynamic @ T_dynamic, assume_a="gen", check_finite=False
    )
    R_static = -pt.linalg.solve(
        B_s_top, D_top + (B_top + C_top @ T_dynamic) @ R_dynamic, assume_a="gen", check_finite=False
    )

    # Assembled in [static | dynamic] order, where the static columns of T are zero, then permuted back.
    T = pt.concatenate(
        [
            pt.concatenate([pt.zeros((n_static, n_static), dtype=T_static.dtype), T_static], axis=1),
            pt.concatenate([pt.zeros((n_dynamic, n_static), dtype=T_dynamic.dtype), T_dynamic], axis=1),
        ],
        axis=0,
    )
    R = pt.concatenate([R_static, R_dynamic], axis=0)

    # The solves against B_s_top skip the finiteness check, so a singular B_s gives finite garbage where the full solve
    # would fail. B_s_top is the triangular QR factor of B_s in value, so its diagonal measures the rank.
    diagonal = pt.abs(pt.diagonal(B_s_top))
    singular = diagonal.min() <= static_mask.size * np.finfo(T.dtype).eps * diagonal.max()
    nan = pt.constant(np.nan, dtype=T.dtype)
    T = pt.switch(singular, nan, T)
    R = pt.switch(singular, nan, R)

    inverse_order = np.argsort(np.concatenate([static_idx, dynamic_idx]))
    if not np.array_equal(inverse_order, np.arange(inverse_order.size)):
        T = T[inverse_order][:, inverse_order]
        R = R[inverse_order]

    return T, R, *extras


class _ConstantOrthogonalBasis(TensorSymbolicOp):
    # ``disconnected_grad`` is a ViewOp, which the canonicalize rewrites delete, and PyMC canonicalizes the logp before
    # differentiating it. The zero gradient is set on the Op instead, since pytensor has no gradient for the full-mode
    # QR of a tall matrix.
    inline = True

    def build_inner_graph(self, x):
        Q, _ = pt.linalg.qr(x, mode="full")
        return [Q]

    def pullback(self, inputs, outputs, output_grads):  # noqa: ARG002
        return [pt.zeros_like(inputs[0])]


_constant_orthogonal_basis = _ConstantOrthogonalBasis()
