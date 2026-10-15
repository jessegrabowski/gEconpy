from typing import Literal, get_args

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.compile.builders import OpFromGraph
from pytensor.graph import Apply, Op
from pytensor.ifelse import ifelse
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.link.numba.dispatch.linalg.solvers.triangular import _solve_triangular
from pytensor.tensor import TensorVariable
from scipy.linalg import solve_triangular

AdjointMethod = Literal["doubling", "qz", "sylvester", "direct"]
ADJOINT_METHODS: tuple[str, ...] = get_args(AdjointMethod)

# The doubling iterate after k steps sums 2^k terms of a series that shrinks like (rho(M) rho(T))^i, so 30 steps cover
# any model whose spectral radii are not within about 1e-8 of one.
DOUBLING_STEPS = 30


def validate_adjoint_method(method: str) -> None:
    if method not in ADJOINT_METHODS:
        expected = ", ".join(repr(name) for name in ADJOINT_METHODS)
        raise ValueError(f"Unknown adjoint_method {method!r}, expected one of {expected}")


def solve_policy_stein(
    A1: TensorVariable,
    C: TensorVariable,
    T: TensorVariable,
    Q: TensorVariable,
    method: AdjointMethod = "doubling",
) -> TensorVariable:
    r"""
    Solve the generalized Stein equation :math:`A_1 S + C^T S T^T = Q` for :math:`S`.

    With :math:`A_1 = (B + C T)^T` this is the equation for the Lagrange multiplier of the first-order policy function
    adjoint.

    Parameters
    ----------
    A1 : TensorVariable
        Left coefficient, :math:`(B + C T)^T`.
    C : TensorVariable
        Jacobian of the system with respect to variables at t+1. Enters the equation transposed.
    T : TensorVariable
        Policy function matrix. Enters the equation transposed.
    Q : TensorVariable
        Right-hand side.
    method : {"doubling", "qz", "sylvester", "direct"}, optional
        Algorithm for the solve. Defaults to ``"doubling"``.

        ``"doubling"`` runs Smith doubling steps on :math:`S - M S T^T = A_1^{-1} Q` with :math:`M = -A_1^{-1} C^T`,
        using matrix products only, so it compiles on every backend. It always runs 30 steps. It returns
        ``nan`` when the series has not converged after 30 steps, which happens when :math:`\rho(M) \rho(T)` is within
        about :math:`10^{-8}` of one, or when the explicitly formed :math:`M` is so far from normal that its powers
        overflow.

        ``"qz"`` reduces the pencil :math:`(A_1, C^T)` with QZ and :math:`T^T` with a complex Schur decomposition, then
        back-substitutes one column at a time. It never inverts :math:`A_1` and has the smallest backward error of any
        method here, but it has no JAX implementation and supports first derivatives only.

        ``"sylvester"`` maps the equation to the Sylvester equation :math:`P S + S R = F` with a Cayley transform and
        solves that with Bartels-Stewart. It is the fastest method here, but :math:`P` is formed by a solve with
        :math:`A_1 - C^T`, so it loses accuracy as that matrix becomes ill-conditioned, and it does so without warning.

        ``"direct"`` solves the vectorized equation :math:`(T \otimes C^T + I \otimes A_1) \operatorname{vec}(S) =
        \operatorname{vec}(Q)` with a dense LU factorization of an :math:`n^2 \times n^2` matrix. It costs
        :math:`O(n^6)` and is kept as a reference implementation only. It is not recommended.

    Returns
    -------
    S : TensorVariable
        Solution of the Stein equation.

    Notes
    -----
    Keep ``"doubling"`` unless there is a reason to change. It runs on every backend, supports higher-order
    derivatives, and when it fails it returns ``nan``, so a failure is visible in the sampler.

    Switch to ``"qz"`` when gradients come back non-finite while the policy function itself is finite, or when the model
    is badly scaled, meaning its variables or equations differ in magnitude by many orders. Steady-state values far
    from one, and levels raised to large powers, both cause this. A condition number of :math:`B + C T` above about
    :math:`10^8` is a warning sign. ``"qz"`` is the most accurate method here and costs about twice as much as
    ``"doubling"``. Do not use it with the JAX backend, which has no QZ, or when second derivatives are needed, for
    example for a Hessian-based optimizer or Laplace approximation.

    Choose ``"sylvester"`` only after checking that it is accurate for the model at hand. It cannot report its own
    error, which grows with the condition number of :math:`A_1 - C^T` and with how far the Cayley-transformed matrices
    are from normal. On the bundled ``sims_2024/nk_complete_taxes.gcn``, where that condition number is about
    :math:`7 \times 10^5`, the relative error stays below :math:`10^{-8}`. On
    ``sims_2024/nk_complete_more_shocks.gcn``, at about :math:`9 \times 10^8`, it reaches :math:`10^{-3}`. The exact
    figure depends on the right-hand side. To check a model, compare it with ``"qz"`` at a representative parameter
    value:

    .. code-block:: python

        import numpy as np
        import pytensor
        import pytensor.tensor as pt

        from gEconpy.solvers.stein import solve_policy_stein

        A, B, C, D = model.linearize_model()
        T, R = model.solve_model()

        symbols = [pt.dmatrix(name) for name in ("A1", "C", "T", "Q")]
        solvers = {
            method: pytensor.function(symbols, solve_policy_stein(*symbols, method=method))
            for method in ("qz", "sylvester")
        }

        Q = np.random.default_rng(0).normal(size=T.shape)
        S_qz, S_sylvester = (solve((B + C @ T).T, C, T, Q) for solve in solvers.values())
        print(np.abs(S_sylvester - S_qz).max() / np.abs(S_qz).max())

    A relative difference below about :math:`10^{-8}` means ``"sylvester"`` is safe near that parameter value. The
    conditioning changes with the parameters, so repeat the check at a few draws from the prior if the sampler will
    explore a wide region.
    """
    validate_adjoint_method(method)

    if method == "doubling":
        return _stein_doubling(A1, C, T, Q)
    if method == "qz":
        return _stein_qz(A1, C, T, Q)
    if method == "sylvester":
        return _stein_sylvester(A1, C, T, Q)
    return _stein_direct(A1, C, T, Q)


def _stein_doubling(A1, C, T, Q):
    X0 = pt.linalg.solve(A1, Q)
    M0 = -pt.linalg.solve(A1, C.T)
    eps = np.finfo(Q.dtype).eps

    def converged(M, N):
        # The next increment M X N is at most ||M||_1 ||N||_1 relative to X, so below eps no later step changes X.
        return pt.abs(M).sum(axis=0).max() * pt.abs(N).sum(axis=0).max() < eps

    def doubling_step(X, M, N):
        return ifelse(converged(M, N), [X, M, N], [X + M @ X @ N, M @ M, N @ N])

    X, M, N = pytensor.scan(doubling_step, outputs_info=[X0, M0, T.T], n_steps=DOUBLING_STEPS, return_updates=False)
    return pt.switch(converged(M[-1], N[-1]), X[-1], pt.constant(np.nan, dtype=Q.dtype))


def _stein_qz(A1, C, T, Q):
    inputs = [A1.type(), C.type(), T.type(), Q.type()]
    A1_inner, C_inner, T_inner, Q_inner = inputs

    AA, BB, Q_left, Z = pt.linalg.qz(A1_inner, C_inner.T, output="complex")
    R, V = pt.linalg.schur(T_inner.T, output="complex")

    F = Q_left.conj().T @ Q_inner @ V
    Y = TriangularStein()(AA, BB, R, F)
    S = pt.real(Z @ Y @ V.conj().T).astype(Q.dtype)

    # QZ, Schur and the back substitution have no gradients. Wrapping them replaces the confusing error pt.real would
    # raise first with one that names the fix.
    solve_op = OpFromGraph(inputs=inputs, outputs=[S], pullback=_no_qz_pullback, name="SteinQZ", inline=True)
    return solve_op(A1, C, T, Q)


def _no_qz_pullback(inputs, outputs, cotangents):
    raise NotImplementedError(
        "adjoint_method='qz' supports first derivatives only. "
        "Use adjoint_method='doubling' or 'sylvester' for higher-order derivatives."
    )


def _stein_sylvester(A1, C, T, Q):
    # With M = A1^{-1} C^T and N = T^T, the Cayley transform takes S + M S N = A1^{-1} Q to P S + S R = F, where
    # P = (I - M)^{-1} (I + M) and R = (I - N) (I + N)^{-1}. Writing P on the pencil keeps A1 itself out of any solve.
    eye = pt.eye(A1.shape[0], dtype=A1.dtype)
    N = T.T
    pencil_difference = A1 - C.T

    P = pt.linalg.solve(pencil_difference, A1 + C.T)
    R = pt.linalg.solve((eye + N).T, (eye - N).T).T
    F = 2 * pt.linalg.solve((eye + N).T, pt.linalg.solve(pencil_difference, Q).T).T

    return pt.linalg.solve_sylvester(P, R, F)


def _stein_direct(A1, C, T, Q):
    # kron(X, Y) vec(S) = vec(Y S X^T) for the column-major vec, and the row-major ravel of S^T is that vec.
    n = A1.shape[0]
    system = pt.linalg.kron(T, C.T) + pt.linalg.kron(pt.eye(n, dtype=A1.dtype), A1)
    vec_S = pt.linalg.solve(system, Q.T.ravel(), assume_a="gen", check_finite=False)
    return vec_S.reshape((n, n)).T


class TriangularStein(Op):
    r"""
    Solve :math:`A Y + B Y R = F` for upper-triangular :math:`A`, :math:`B` and :math:`R`.

    Column :math:`j` of :math:`Y` solves the triangular system
    :math:`(A + R_{jj} B) y_j = f_j - B \sum_{k<j} y_k R_{kj}`, so the columns are found in order by back substitution.
    """

    __props__ = ()
    gufunc_signature = "(m,m),(m,m),(n,n),(m,n)->(m,n)"

    def make_node(self, A, B, R, F) -> Apply:
        inputs = list(map(pt.as_tensor, [A, B, R, F]))
        outputs = [pt.tensor("Y", dtype=inputs[3].type.dtype, shape=inputs[3].type.shape)]
        return Apply(self, inputs, outputs)

    def infer_shape(self, node, input_shapes):  # noqa: ARG002
        return [input_shapes[3]]

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:  # noqa: ARG002
        A, B, R, F = inputs
        Y = np.zeros_like(F)

        # A singular column system means the Stein equation has no unique solution. Return nan, as the Numba
        # implementation does, so a sampler rejects the draw instead of aborting.
        try:
            for j in range(F.shape[1]):
                rhs = F[:, j] - B @ (Y[:, :j] @ R[:j, j])
                Y[:, j] = solve_triangular(A + R[j, j] * B, rhs, lower=False, check_finite=False)
        except np.linalg.LinAlgError:
            Y[:] = np.nan

        outputs[0][0] = Y


@register_funcify_default_op_cache_key(TriangularStein)
def numba_funcify_TriangularStein(op, node, **kwargs):  # noqa: ARG001
    """Run the column back substitution as one compiled loop."""

    @numba_basic.numba_njit
    def triangular_stein(A, B, R, F):
        # Columns of Y, R and F are stored as rows so every slice in the loop is contiguous.
        F_columns = np.ascontiguousarray(F.T)
        R_columns = np.ascontiguousarray(R.T)
        Y_columns = np.zeros_like(F_columns)

        for j in range(F_columns.shape[0]):
            rhs = F_columns[j].copy()
            if j > 0:
                rhs -= B @ (R_columns[j, :j] @ Y_columns[:j])
            Y_columns[j] = _solve_triangular(A + R_columns[j, j] * B, rhs, 0, False, False, True)

        return np.ascontiguousarray(Y_columns.T)

    cache_version = 1
    return triangular_stein, cache_version
