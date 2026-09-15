import logging

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.compile.builders import OpFromGraph
from pytensor.graph import Apply, Op
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.link.numba.dispatch.linalg.decomposition.lu_factor import _lu_factor
from pytensor.link.numba.dispatch.linalg.solvers.general import _solve_gen
from pytensor.link.numba.dispatch.linalg.solvers.lu_solve import _getrs
from pytensor.tensor import TensorVariable
from pytensor.tensor.linalg.dtype_utils import linalg_output_dtype

from gEconpy.solvers.shared import (
    o1_policy_function_adjoints,
    pt_compute_selection_matrix,
    stabilize,
)

_log = logging.getLogger(__name__)

_CONVERGED_MESSAGE = "Optimization successful"
_NOT_CONVERGED_MESSAGE = "Iteration on all matrices failed to converge"


def cycle_reduction_numpy(
    A0: np.ndarray,
    A1: np.ndarray,
    A2: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> tuple[np.ndarray | None, np.ndarray | None, str, float]:
    """
    Solve the matrix quadratic equation :math:`A_0 + A_1 X + A_2 X^2 = 0` by the cycle reduction algorithm of [1]_.

    In the DSGE context the solution is the first-order policy function :math:`T`. Adapted from
    https://github.com/DynareTeam/dynare/blob/master/matlab/cycle_reduction.m.

    Parameters
    ----------
    A0 : ndarray
        Constant coefficient of the matrix quadratic. In a DSGE model this is the Jacobian with respect to lagged
        variables.
    A1 : ndarray
        Linear coefficient of the matrix quadratic. In a DSGE model this is the Jacobian with respect to current
        variables.
    A2 : ndarray
        Quadratic coefficient of the matrix quadratic. In a DSGE model this is the Jacobian with respect to variables
        that enter in expectation.
    max_iter : int, optional
        Maximum number of iterations before giving up. Defaults to 1000.
    tol : float, optional
        Floating point tolerance used to detect convergence. Defaults to 1e-7.

    Returns
    -------
    X : ndarray or None
        Solution of the matrix quadratic equation, or None when the iteration did not converge.
    res : ndarray or None
        Residual :math:`A_0 + A_1 X + A_2 X^2` at the solution, or None when the iteration did not converge.
    result : str
        ``"Optimization successful"`` on convergence, otherwise a message naming the failure.
    log_norm : float
        Logarithm of the L1 norm of ``A1`` at the final iteration when the iteration did not converge, otherwise 0.

    References
    ----------
    .. [1] Bini, D.A., Latouche, G., and Meini, B. "Solving matrix polynomial equations
       arising in queueing problems." *Linear Algebra and its Applications* 340 (2002): 222-244.
    """
    # The loop rebinds A0, A1, and A2 to fresh arrays without mutating them in place, so these snapshots can alias
    # the inputs.
    A0_initial = A0
    A1_hat = A1
    A1_initial = A1
    A2_initial = A2

    n, _ = A0.shape
    idx_0 = np.arange(n)
    idx_1 = idx_0 + n

    for i in range(int(max_iter)):
        tmp = np.vstack((A0, A2)) @ np.linalg.solve(A1, np.hstack((A0, A2)))

        A1 = A1 - tmp[idx_0, :][:, idx_1] - tmp[idx_1, :][:, idx_0]
        A0 = -tmp[idx_0, :][:, idx_0]
        A2 = -tmp[idx_1, :][:, idx_1]
        A1_hat = A1_hat - tmp[idx_1, :][:, idx_0]

        A0_L1_norm = np.linalg.norm(A0, ord=1)
        if A0_L1_norm < tol:
            A2_L1_norm = np.linalg.norm(A2, ord=1)
            if A2_L1_norm < tol:
                break

        elif np.isnan(A0_L1_norm) or i == (max_iter - 1):
            log_norm = np.log(np.linalg.norm(A1, 1))
            return None, None, _NOT_CONVERGED_MESSAGE, log_norm

    X = -np.linalg.solve(A1_hat, A0_initial)
    res = A0_initial + A1_initial @ X + A2_initial @ X @ X

    return X, res, _CONVERGED_MESSAGE, 0


def solve_policy_function_with_cycle_reduction(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
    verbose: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None, str, float]:
    """
    Solve for the policy function of a linearized DSGE system by cycle reduction.

    Returns the transition matrix T and the selection matrix R, which together define the linear state space
    representation of the model.

    Parameters
    ----------
    A : ndarray
        Jacobian of the system with respect to variables at t-1, evaluated at the steady state.
    B : ndarray
        Jacobian of the system with respect to variables at t, evaluated at the steady state.
    C : ndarray
        Jacobian of the system with respect to variables at t+1, evaluated at the steady state.
    D : ndarray
        Jacobian of the system with respect to exogenous shocks, evaluated at the steady state.
    max_iter : int, optional
        Maximum number of iterations before giving up. Defaults to 100.
    tol : float, optional
        Floating point tolerance used to detect convergence. Defaults to 1e-8.
    verbose : bool, optional
        Log the sum of squared residuals at the solution, or the failure message when the iteration did not converge.
        Defaults to True.

    Returns
    -------
    T : ndarray or None
        Transition matrix, giving the effect of variable values at t on their values at t+1. None when the iteration
        did not converge.
    R : ndarray or None
        Selection matrix, giving the effect of exogenous shocks at t on variable values at t+1. None when the
        iteration did not converge.
    result : str
        Message describing the outcome of the cycle reduction iteration.
    log_norm : float
        Log L1 norm of ``A1`` at the final iteration when the iteration did not converge, otherwise 0.
    """
    T, res, result, log_norm = cycle_reduction_numpy(A, B, C, max_iter, tol)

    if T is None:
        if verbose:
            _log.info(
                f"Solution not found. Solver returned: {result}\n"
                f"Log norm of the solution at the final iteration: {log_norm:0.9f}"
            )
        return None, None, result, log_norm

    if verbose:
        _log.info(f"Solution found, sum of squared residuals: {(res**2).sum():0.9f}")

    T = np.ascontiguousarray(T)
    R = -np.linalg.solve(C @ T + B, D)

    return T, R, result, log_norm


def cycle_reduction_pt(
    A: TensorVariable,
    B: TensorVariable,
    C: TensorVariable,
    D: TensorVariable,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> tuple[TensorVariable, TensorVariable]:
    """
    Build a symbolic graph that solves a linearized DSGE system by cycle reduction.

    The cycle reduction iteration runs inside an Op, so the graph itself holds a single node.

    Parameters
    ----------
    A : TensorVariable
        Jacobian of the system with respect to variables at t-1, evaluated at the steady state.
    B : TensorVariable
        Jacobian of the system with respect to variables at t, evaluated at the steady state.
    C : TensorVariable
        Jacobian of the system with respect to variables at t+1, evaluated at the steady state.
    D : TensorVariable
        Jacobian of the system with respect to exogenous shocks, evaluated at the steady state.
    max_iter : int, optional
        Maximum number of cycle reduction iterations. Defaults to 1000.
    tol : float, optional
        Floating point tolerance used to detect convergence. Defaults to 1e-9.

    Returns
    -------
    T : TensorVariable
        Transition matrix, giving the effect of variable values at t on their values at t+1.
    R : TensorVariable
        Selection matrix, giving the effect of exogenous shocks at t on variable values at t+1.
    """
    T = CycleReductionWrapper(max_iter=max_iter, tol=tol)(A, B, C)
    R = pt_compute_selection_matrix(B, C, D, T)
    return T, R


def scan_cycle_reduction(
    A: pt.TensorLike,
    B: pt.TensorLike,
    C: pt.TensorLike,
    D: pt.TensorLike,
    max_iter: int = 50,
    tol: float = 1e-7,
    use_adjoint_gradients: bool = True,
) -> tuple[TensorVariable, TensorVariable, TensorVariable]:
    """
    Build a symbolic graph that solves a linearized DSGE system with an unrolled cycle reduction scan.

    The iteration is a pytensor scan built from ordinary tensor Ops, so it compiles on every backend and can be
    differentiated directly. :func:`~gEconpy.solvers.cycle_reduction.cycle_reduction_pt` wraps the iteration in an Op
    with a numba dispatch only.

    Parameters
    ----------
    A : TensorVariable
        Jacobian of the system with respect to variables at t-1, evaluated at the steady state.
    B : TensorVariable
        Jacobian of the system with respect to variables at t, evaluated at the steady state.
    C : TensorVariable
        Jacobian of the system with respect to variables at t+1, evaluated at the steady state.
    D : TensorVariable
        Jacobian of the system with respect to exogenous shocks, evaluated at the steady state.
    max_iter : int, optional
        Number of scan steps. Steps taken after convergence are no-ops. Defaults to 50.
    tol : float, optional
        Floating point tolerance used to detect convergence. Defaults to 1e-7.
    use_adjoint_gradients : bool, optional
        When True, differentiate with the closed-form adjoints of the matrix quadratic equation. When False,
        backpropagate through the scan. Defaults to True.

    Returns
    -------
    T : TensorVariable
        Transition matrix, giving the effect of variable values at t on their values at t+1.
    R : TensorVariable
        Selection matrix, giving the effect of exogenous shocks at t on variable values at t+1.
    n_steps : TensorVariable
        Number of cycle reduction steps taken before convergence.
    """
    A = pt.as_tensor_variable(A, name="A")
    B = pt.as_tensor_variable(B, name="B")
    C = pt.as_tensor_variable(C, name="C")
    D = pt.as_tensor_variable(D, name="D")

    output = _scan_cycle_reduction(A, B, C, max_iter, tol)

    scan_cycle_reduction_op = OpFromGraph(
        inputs=[A, B, C],
        outputs=output,
        pullback=_policy_function_pullback if use_adjoint_gradients else None,
        name="ScanCycleReduction",
        inline=True,
    )

    T, n_steps = scan_cycle_reduction_op(A, B, C)
    R = pt_compute_selection_matrix(B, C, D, T)

    return T, R, n_steps


class CycleReductionWrapper(Op):
    """
    PyTensor Op that solves the matrix quadratic :math:`A + B T + C T^2 = 0` by cycle reduction.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of cycle reduction iterations. Defaults to 1000.
    tol : float, optional
        Floating point tolerance used to detect convergence. Defaults to 1e-9.
    """

    __props__ = ("max_iter", "tol")
    gufunc_signature = "(n,n),(n,n),(n,n)->(n,n)"

    def __init__(self, max_iter: int = 1000, tol: float = 1e-9):
        self.max_iter = int(max_iter)
        self.tol = tol
        super().__init__()

    def make_node(self, A, B, C) -> Apply:
        inputs = list(map(pt.as_tensor, [A, B, C]))
        o_dtype = linalg_output_dtype(*(inp.type.dtype for inp in inputs))
        outputs = [pt.tensor("T", dtype=o_dtype, shape=inputs[0].type.shape)]

        return Apply(self, inputs, outputs)

    def infer_shape(self, node, input_shapes):
        n = input_shapes[0][0]
        return [(n, n)]

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        A, B, C = inputs
        T, _res, _result, _log_norm = cycle_reduction_numpy(A, B, C, max_iter=self.max_iter, tol=self.tol)

        outputs[0][0] = np.asarray(T, dtype=node.outputs[0].type.dtype)

    def pullback(self, inputs, outputs, cotangents):
        return _policy_function_pullback(inputs, outputs, cotangents)


@register_funcify_default_op_cache_key(CycleReductionWrapper)
def numba_funcify_CycleReductionWrapper(op, node, **kwargs):  # noqa: ARG001
    """Route :class:`CycleReductionWrapper` to the njit kernel, casting inputs to the working dtype when they differ."""
    max_iter = op.max_iter
    tol = op.tol

    out_dtype = node.outputs[0].type.numpy_dtype
    must_cast_A, must_cast_B, must_cast_C = (inp.type.numpy_dtype != out_dtype for inp in node.inputs)

    @numba_basic.numba_njit
    def cycle_reduction(A, B, C):
        if must_cast_A:
            A = A.astype(out_dtype)
        if must_cast_B:
            B = B.astype(out_dtype)
        if must_cast_C:
            C = C.astype(out_dtype)
        T, _converged = _cycle_reduction_core(A, B, C, max_iter, tol)
        return T

    cache_version = 3
    return cycle_reduction, cache_version


def _policy_function_pullback(inputs, outputs, output_grads):
    # CycleReductionWrapper exposes a single output (T) and the scan OpFromGraph exposes two (T, n_steps). Only T
    # carries gradient in either case.
    A, B, C = inputs
    T = outputs[0]
    T_bar = output_grads[0]

    return o1_policy_function_adjoints(A, B, C, T, T_bar)


@numba_basic.numba_njit(final_function=True)
def _cycle_reduction_core(
    A0: np.ndarray, A1: np.ndarray, A2: np.ndarray, max_iter: int, tol: float
) -> tuple[np.ndarray, bool]:
    n = A0.shape[0]
    dtype = A0.dtype

    # Inputs are read-only (pytensor never grants overwrite here), so the loop rebinds to fresh arrays and these
    # aliases stay valid.
    A0_initial = A0
    A1_hat = A1

    # Owned scratch, overwritten freely. lu_buf and rhs* need Fortran order, and numba rejects order="F", hence the
    # `.T` of a fresh C array.
    m00 = np.empty((n, n), dtype=dtype)
    m02 = np.empty((n, n), dtype=dtype)
    m20 = np.empty((n, n), dtype=dtype)
    m22 = np.empty((n, n), dtype=dtype)
    lu_buf = np.empty((n, n), dtype=dtype).T
    rhs0 = np.empty((n, n), dtype=dtype).T
    rhs2 = np.empty((n, n), dtype=dtype).T

    converged = False
    for _ in range(int(max_iter)):
        # One LU of A1 serves both solves. `_lu_factor` returns 0-based pivots but `_getrs` hands them straight to
        # LAPACK, which expects 1-based, so shift them here.
        lu_buf[:] = A1
        lu, piv = _lu_factor(lu_buf, True)
        piv += np.int32(1)

        rhs0[:] = A0
        A1_inv_A0, _info0 = _getrs(lu, rhs0, piv, 0, True)
        rhs2[:] = A2
        A1_inv_A2, _info2 = _getrs(lu, rhs2, piv, 0, True)

        np.dot(A0, A1_inv_A0, m00)
        np.dot(A0, A1_inv_A2, m02)
        np.dot(A2, A1_inv_A0, m20)
        np.dot(A2, A1_inv_A2, m22)

        A1 = A1 - m02 - m20
        A1_hat = A1_hat - m20
        A0 = -m00
        A2 = -m22

        A0_norm = np.linalg.norm(A0, ord=1)
        if A0_norm < tol:
            if np.linalg.norm(A2, ord=1) < tol:
                converged = True
                break
        elif np.isnan(A0_norm):
            break

    # `_solve_gen` does not raise on failure. It NaN-fills, and the caller rejects the draw. A1_hat is owned and may
    # be overwritten. A0_initial aliases an input and must not be.
    T = -_solve_gen(A1_hat, A0_initial, False, True, False, False) if converged else np.zeros_like(A0_initial)

    return T, converged


def _scan_cycle_reduction(A, B, C, max_iter: int = 1000, tol: float = 1e-7) -> list[TensorVariable]:
    def noop(A0, A1, A2, A1_hat, norm, step_num):
        return A0, A1, A2, A1_hat, norm, step_num

    def cycle_step(A0, A1, A2, A1_hat, step_num, idx_0, idx_1):
        tmp = pt.dot(
            pt.vertical_stack(A0, A2),
            pt.linalg.solve(
                stabilize(A1),
                pt.horizontal_stack(A0, A2),
                assume_a="gen",
                check_finite=False,
            ),
        )

        A1 = A1 - tmp[idx_0, :][:, idx_1] - tmp[idx_1, :][:, idx_0]
        A0 = -tmp[idx_0, :][:, idx_0]
        A2 = -tmp[idx_1, :][:, idx_1]
        A1_hat = A1_hat - tmp[idx_1, :][:, idx_0]

        A0_L1_norm = pt.linalg.norm(A0, ord=1)

        return A0, A1, A2, A1_hat, A0_L1_norm, step_num + 1

    def step(A0, A1, A2, A1_hat, norm, step_num, idx_0, idx_1, tol):
        return pytensor.ifelse(
            norm < tol,
            noop(A0, A1, A2, A1_hat, norm, step_num),
            cycle_step(A0, A1, A2, A1_hat, step_num, idx_0, idx_1),
        )

    n = A.shape[0]
    idx_0 = pt.arange(n)
    idx_1 = idx_0 + n
    norm = np.array(1e9, dtype="float64")
    step_num = pt.zeros((), dtype="int32")
    *_, A1_hat, norm, n_steps = pytensor.scan(
        step,
        outputs_info=[A, B, C, B, norm, step_num],
        non_sequences=[idx_0, idx_1, tol],
        n_steps=max_iter,
        return_updates=False,
    )
    A1_hat = A1_hat[-1]

    T = -pt.linalg.solve(stabilize(A1_hat), A, assume_a="gen", check_finite=False)

    return [T, n_steps[-1]]
