import numba as nb
import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.compile import get_mode
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import Apply, Op

from gEconpy.model.perturbation import _log
from gEconpy.solvers.shared import (
    o1_policy_function_adjoints,
    pt_compute_selection_matrix,
    stabilize,
)


# TODO: These njit decorators cause the CI to fail on Windows only -- no idea why. Disabling it for now.
# @nb.njit(
#     (nb.float64[:, ::1], nb.float64[:, ::1], nb.float64[:, ::1], nb.int64, nb.float64),
#     cache=True,
# )
def nb_cycle_reduction(
    A0: np.ndarray,
    A1: np.ndarray,
    A2: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> tuple[np.ndarray | None, np.ndarray | None, str, float]:
    """
    Solve quadratic matrix equation of the form $A0x^2 + A1x + A2 = 0$ via cycle reduction algorithm of [1].
    Useful in the DSGE context to solve for the implicit derivative of the policy function, g, with respect to
    state vector y.

    Adapted from the Dynare file cycle_reduction.m, found at
    https://github.com/DynareTeam/dynare/blob/master/matlab/cycle_reduction.m

    Parameters
    ----------
    A0: Arraylike
        Coefficient matrix associated with the constant term of the matrix quadratic equation. In DSGE models, this is
        dF_d_t-1, the derivative of the system with respect to variables that enter as lags
    A1: ArrayLike
        Coefficient matrix associated with the linear term of the matrix quadratic equation. In DSGE models, this is
        dF_d_t, the derivative of the system with respect to variables that enter at the current time
    A2: ArrayLike
        Coefficient matrix associated with the quadratic term of the matrix quadratic equation. In DSGE models, this is
        dF_d_t+1, the derivative of the system with respect to variables that enter in expectation
    max_iter: int, default: 1000
        Maximum number of iterations to perform before giving up.
    tol: float, default: 1e-7
        Floating point tolerance used to detect algorithmic convergence

    Returns
    -------
    X: array
        Solution to matrix quadratic equation
    res: array
        Residual of the matrix quadratic equation, or None if the algorithm fails to converge
    result: str
        String indicating the result of the optimization. If the algorithm converges, this will be "Optimization
        successful". If the algorithm fails to converge, this will be "Iteration on all matrices failed to converged"
    log_norm: float
        Logarithm of the L1 norm of the matrix A1. This is useful for diagnosing the success of the algorithm.

    References
    ----------
    ..[1] D.A. Bini, G. Latouche, B. Meini (2002), "Solving matrix polynomial equations
          arising in queueing problems", Linear Algebra and its Applications 340, pp. 222-244
    ..[2]

    """
    result = "Optimization successful"
    log_norm = 0
    X = None
    res = None

    A0_initial = A0.copy()
    A1_hat = A1.copy()

    A1_initial = A1.copy()
    A2_initial = A2.copy()

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
            # Algorithm is successful when the L1 norm of A2 is sufficiently small
            A2_L1_norm = np.linalg.norm(A2, ord=1)
            if A2_L1_norm < tol:
                break

        elif np.isnan(A0_L1_norm) or i == (max_iter - 1):
            # If we fail, figure out how far we got
            if A0_L1_norm < tol:
                result = "Iteration on matrix A0 and A1 converged towards a solution, but A2 did not."
                log_norm = np.log(np.linalg.norm(A2, 1))
            else:
                result = "Iteration on all matrices failed to converged"
                log_norm = np.log(np.linalg.norm(A1, 1))

            return X, res, result, log_norm

    X = -np.linalg.solve(A1_hat, A0_initial)
    res = A0_initial + A1_initial @ X + A2_initial @ X @ X

    return X, res, result, log_norm


# @nb.njit(
#     (nb.float64[:, ::1], nb.float64[:, ::1], nb.float64[:, ::1], nb.float64[:, ::1]),
#     cache=True,
# )
def nb_solve_shock_matrix(B: np.ndarray, C: np.ndarray, D: np.ndarray, G_1: np.ndarray):
    """
    Given the partial solution to the linear approximate policy function G_1, solve for the remaining component of the
    policy function, R.

    Parameters
    ----------
    B: ArrayLike
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to variables that
        are observed when decision making: those with t subscripts.
    C: Arraylike
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to variables that
        enter in expectation when decision making: those with t+1 subscripts.
    D: ArrayLike
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to exogenous shocks.
    G_1: ArrayLike
        Transition matrix T in state space jargon. Gives the effect of variable values at time t on the
        values of the variables at time t+1.

    Returns
    -------
    impact: ArrayLike
        Selection matrix R in state space jargon. Gives the effect of exogenous shocks at time t on the values of
        system variables at time t+1.

    """

    return -np.linalg.solve(C @ G_1 + B, D.astype(C.dtype))


def _linear_policy_jvp(inputs, outputs, output_grads):
    A, B, C = inputs
    [T] = outputs
    [T_bar] = output_grads

    return o1_policy_function_adjoints(A, B, C, T, T_bar)


class CycleReductionWrapper(Op):
    def __init__(self, max_iter=1000, tol=1e-9):
        self.max_iter = int(max_iter)
        self.tol = tol
        super().__init__()

    def make_node(self, A, B, C) -> Apply:
        inputs = list(map(pt.as_tensor, [A, B, C]))
        outputs = [pt.dmatrix("T")]

        return Apply(self, inputs, outputs)

    def perform(
        self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]
    ) -> None:
        A, B, C = inputs
        T, res, result, log_norm = nb_cycle_reduction(
            A, B, C, max_iter=self.max_iter, tol=self.tol
        )

        outputs[0][0] = np.asarray(T)

    def L_op(self, inputs, outputs, output_grads):
        return _linear_policy_jvp(inputs, outputs, output_grads)


def cycle_reduction_pt(A, B, C, D, max_iter=1000, tol=1e-9):
    T = CycleReductionWrapper(max_iter=max_iter, tol=tol)(A, B, C)
    R = pt_compute_selection_matrix(B, C, D, T)
    return T, R


def _scan_cycle_reduction(
    A, B, C, max_iter: int = 1000, tol: float = 1e-7, mode=None
) -> pt.Variable:
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
        state = pytensor.ifelse(
            norm < tol,
            noop(A0, A1, A2, A1_hat, norm, step_num),
            cycle_step(A0, A1, A2, A1_hat, step_num, idx_0, idx_1),
        )
        return state

    n = A.shape[0]
    idx_0 = pt.arange(n)
    idx_1 = idx_0 + n
    norm = np.array(1e9, dtype="float64")
    step_num = pt.zeros((), dtype="int32")
    (*_, A1_hat, norm, n_steps), updates = pytensor.scan(
        step,
        outputs_info=[A, B, C, B, norm, step_num],
        non_sequences=[idx_0, idx_1, tol],
        n_steps=max_iter,
        mode=get_mode(mode),
    )
    A1_hat = A1_hat[-1]

    T = -pt.linalg.solve(stabilize(A1_hat), A, assume_a="gen", check_finite=False)

    return [T, n_steps[-1]]


def scan_cycle_reduction(
    A: pt.TensorLike,
    B: pt.TensorLike,
    C: pt.TensorLike,
    D: pt.TensorLike,
    max_iter: int = 50,
    tol: float = 1e-7,
    mode: str | None = None,
    use_adjoint_gradients: bool = True,
):
    A = pt.as_tensor_variable(A, name="A")
    B = pt.as_tensor_variable(B, name="B")
    C = pt.as_tensor_variable(C, name="C")
    D = pt.as_tensor_variable(D, name="D")

    output = _scan_cycle_reduction(A, B, C, max_iter, tol, mode=mode)

    ScanCycleReducation = OpFromGraph(
        inputs=[A, B, C],
        outputs=output,
        lop_overrides=_linear_policy_jvp if use_adjoint_gradients else None,
        name="ScanCycleReduction",
        inline=True,
    )

    T, n_steps = ScanCycleReducation(A, B, C)
    R = pt_compute_selection_matrix(B, C, D, T)

    return T, R, n_steps


def solve_policy_function_with_cycle_reduction(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """
    Solve quadratic matrix equation of the form $A0x^2 + A1x + A2 = 0$ via cycle reduction algorithm of [1] to
    obtain the first-order linear approxiate policy matrices T and R.

    Parameters
    ----------
    A: np.ndarray
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to past variables
        values that are known when decision-making: those with t-1 subscripts.
    B: np.ndarray
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to variables that
        are observed when decision-making: those with t subscripts.
    C: np.ndarray
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to variables that
        enter in expectation when decision-making: those with t+1 subscripts.
    D: np.ndarray
        Jacobian matrix of the DSGE system, evaluated at the steady state, taken with respect to exogenous shocks.
    max_iter: int, default: 1000
        Maximum number of iterations to perform before giving up.
    tol: float, default: 1e-7
        Floating point tolerance used to detect algorithmic convergence
    verbose: bool, default: True
        If true, prints the sum of squared residuals that result when the system is computed used the solution.

    Returns
    -------
    T: ArrayLike
        Transition matrix T in state space jargon. Gives the effect of variable values at time t on the
        values of the variables at time t+1.
    R: ArrayLike
        Selection matrix R in state space jargon. Gives the effect of exogenous shocks at the t on the values of
        variables at time t+1.
    result: str
        String describing result of the cycle reduction algorithm
    log_norm: float
        Log L1 matrix norm of the first matrix (A2 -> A1 -> A0) that did not converge.
    """

    # Sympy gives back integers in the case of x/dx = 1, which can screw up the dtypes when passing to numba if
    # a Jacobian matrix is all constants (i.e. dF/d_shocks) -- cast everything to float64 here to avoid
    # a numba warning.
    T, R = None, None
    T, res, result, log_norm = nb_cycle_reduction(A, B, C, max_iter, tol)
    T = np.ascontiguousarray(T)

    if verbose:
        if result == "Optimization successful":
            _log.info(
                f"Solution found, sum of squared residuals: {(res**2).sum():0.9f}",
            )
        else:
            _log.info(
                f"Solution not found. Solver returned: {result}\n,"
                f"Log norm of the solution at the final iteration: {log_norm:0.9f}"
            )

    if T is not None:
        R = nb_solve_shock_matrix(B, C, D, T)

    return T, R, result, log_norm
