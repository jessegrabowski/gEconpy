import numpy as np
import pytensor.tensor as pt

from numba.extending import register_jitable
from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.tensor import TensorLike, TensorVariable
from pytensor.tensor.symbolic import TensorSymbolicOp


def generalized_eig(a: TensorLike, b: TensorLike, *, tol: float) -> tuple[TensorVariable, TensorVariable]:
    r"""
    Compute the generalized eigenvalues of the real pencil :math:`a x = \lambda b x`.

    The eigenvalues come from the real QZ decomposition :func:`pytensor.tensor.linalg.qz`, the same as
    :func:`scipy.linalg.eig` with ``b`` given, and are returned as real tensors, so reverse-mode differentiation
    through both components works. They are sorted by ascending modulus.

    A finite simple eigenvalue has the derivative

    .. math::

        d\lambda = \frac{y^H (da - \lambda \, db) x}{y^H b x},

    with right and left eigenvectors :math:`x` and :math:`y`. The pullback computes them from a complex QZ by
    back-substitution with :class:`SchurEigenvectors`. Infinite eigenvalues get a zero gradient. Only first-order
    derivatives are supported.

    Parameters
    ----------
    a, b : TensorVariable
        Real-valued square matrices of shape ``(n, n)``.
    tol : float
        An eigenvalue is infinite when its denominator ``beta`` from the QZ decomposition has modulus at most
        ``tol``. Infinite eigenvalues have real part ``inf`` and imaginary part zero.

    Returns
    -------
    eigvals_real : TensorVariable
        Real parts of the eigenvalues, shape ``(n,)``.
    eigvals_imag : TensorVariable
        Imaginary parts of the eigenvalues, shape ``(n,)``.

    Examples
    --------
    Differentiate the moduli of the finite eigenvalues of a pencil with respect to both matrices:

    .. code-block:: python

        import pytensor.tensor as pt
        from gEconpy.pytensorf.generalized_eig import generalized_eig

        a = pt.dmatrix("a")
        b = pt.dmatrix("b")
        re, im = generalized_eig(a, b, tol=1e-8)
        modulus = pt.sqrt(re**2 + im**2)
        grad_a, grad_b = pt.grad(modulus[pt.isfinite(modulus)].sum(), [a, b])
    """
    return GeneralizedEig(tol=tol)(a, b)


class GeneralizedEig(TensorSymbolicOp):
    """
    PyTensor Op whose inner graph computes sorted generalized eigenvalues from a real :func:`pytensor.tensor.linalg.qz`.

    The pullback is described in :func:`generalized_eig`.

    Parameters
    ----------
    tol : float
        Modulus of the QZ denominator ``beta`` at or below which an eigenvalue is infinite.
    """

    __props__ = ("tol",)
    inline = True

    def __init__(self, *, tol: float, **kwargs):
        self.tol = float(tol)
        super().__init__(**kwargs)

    def build_inner_graph(self, a, b):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                f"generalized_eig requires two square matrices, but got inputs with ndim={a.ndim} and ndim={b.ndim}."
            )
        return list(_sorted_real_qz_eigenvalues(a, b, tol=self.tol))

    def pullback(self, inputs, outputs, output_grads):
        a, b = inputs
        eigvals_real, eigvals_imag = outputs
        g_real, g_imag = output_grads

        if isinstance(g_real.type, DisconnectedType):
            g_real = pt.zeros_like(eigvals_real)
        if isinstance(g_imag.type, DisconnectedType):
            g_imag = pt.zeros_like(eigvals_imag)

        S, T, Q, Z = pt.linalg.qz(a, b, output="complex")
        X, Y = SchurEigenvectors()(S, T, Q, Z, eigvals_real, eigvals_imag)

        g_conj = g_real - 1j * g_imag
        infinite = pt.isinf(eigvals_real)
        eigvals = pt.switch(infinite, 0.0, eigvals_real) + 1j * pt.switch(infinite, 0.0, eigvals_imag)

        Y_conj = Y.conj()
        a_bar = ((Y_conj * g_conj[None, :]) @ X.T).real
        b_bar = -((Y_conj * (g_conj * eigvals)[None, :]) @ X.T).real

        return [a_bar.astype(a.dtype), b_bar.astype(b.dtype)]


def _sorted_real_qz_eigenvalues(a, b, tol):
    *_, alpha, beta, _, _ = pt.linalg.qz(a, b, output="real", return_eigenvalues=True)

    finite = pt.abs(beta) > tol
    ratio = alpha / pt.switch(finite, beta, 1.0)
    eigvals_real = pt.switch(finite, pt.real(ratio), np.inf)
    eigvals_imag = pt.switch(finite, pt.imag(ratio), 0.0)
    order = pt.argsort(eigvals_real**2 + eigvals_imag**2, kind="mergesort")

    return eigvals_real[order], eigvals_imag[order]


@register_jitable
def _match_schur_diagonal(s, t, eigvals):
    """Index of the unused nonzero Schur diagonal pair ``(s_j, t_j)`` closest to each finite eigenvalue, or -1."""
    n = s.shape[0]
    used = np.zeros(n, dtype=np.bool_)
    match = np.full(n, -1, dtype=np.int64)
    pair_norm = np.sqrt(np.abs(s) ** 2 + np.abs(t) ** 2)
    degenerate = pair_norm == 0.0
    pair_norm[degenerate] = 1.0

    for k in range(n):
        if not np.isfinite(eigvals[k].real):
            continue
        distance = np.abs(s - eigvals[k] * t) / (pair_norm * np.sqrt(1.0 + np.abs(eigvals[k]) ** 2))
        distance[used | degenerate] = np.inf
        match[k] = np.argmin(distance)
        used[match[k]] = True

    return match


@register_jitable
def _schur_eigenvectors(S, T, Q, Z, eigvals_real, eigvals_imag):
    n = S.shape[0]
    X = np.zeros((n, n), dtype=np.complex128)
    Y = np.zeros((n, n), dtype=np.complex128)
    S_scale = np.abs(S).max()
    T_scale = np.abs(T).max()
    eps = np.finfo(np.float64).eps
    match = _match_schur_diagonal(np.diag(S).copy(), np.diag(T).copy(), eigvals_real + 1j * eigvals_imag)

    for column in range(n):
        i = match[column]
        if i < 0:
            continue

        s_i = S[i, i]
        t_i = T[i, i]
        M = t_i * S - s_i * T
        small = eps * (np.abs(t_i) * S_scale + np.abs(s_i) * T_scale)

        v = np.zeros(n, dtype=np.complex128)
        v[i] = 1.0
        for k in range(i - 1, -1, -1):
            if np.abs(M[k, k]) > small:
                v[k] = -np.sum(M[k, k + 1 : i + 1] * v[k + 1 : i + 1]) / M[k, k]

        M_H = np.conj(M.T)
        w = np.zeros(n, dtype=np.complex128)
        w[i] = 1.0
        for k in range(i + 1, n):
            if np.abs(M_H[k, k]) > small:
                w[k] = -np.sum(M_H[k, i:k] * w[i:k]) / M_H[k, k]

        X[:, column] = (Z @ v) / t_i
        Y[:, column] = Q @ w

    # Back-substitution normalizes each pair, but pairs within a repeated eigenvalue are not yet biorthogonal.
    biorthogonality = Y.conj().T @ (Q @ T @ Z.conj().T) @ X
    for column in range(n):
        if match[column] < 0:
            biorthogonality[column, column] = 1.0
    Y = np.linalg.solve(biorthogonality, Y.conj().T).conj().T

    return X, Y


class SchurEigenvectors(Op):
    r"""
    PyTensor Op that computes the eigenvectors of a pencil from its complex QZ factors by back-substitution.

    Given :math:`a = Q S Z^H` and :math:`b = Q T Z^H` with :math:`S` and :math:`T` upper triangular, eigenvalue
    :math:`i` makes the :math:`i`-th diagonal entry of :math:`M = t_i S - s_i T` zero. The right eigenvector is
    :math:`x = Z v / t_i` with :math:`M v = 0`, :math:`v_i = 1` and :math:`v_k = 0` for :math:`k > i`. The left
    eigenvector is :math:`y = Q w` with :math:`w^H M = 0`, :math:`w_i = 1` and :math:`w_k = 0` for :math:`k < i`.
    These supports give :math:`y^H b x = 1`. A diagonal entry of :math:`M` at roundoff level belongs to an eigenvalue
    equal to eigenvalue :math:`i`, and the matching component is set to zero, which picks one eigenvector of a
    semisimple repeated eigenvalue. The left eigenvectors are then rescaled so that :math:`Y^H b X = I`, which makes
    the eigenvectors of a repeated eigenvalue biorthogonal. Column :math:`k` belongs to the given eigenvalue
    :math:`k`, matched to the unused diagonal pair closest to it in chordal distance, and is zero for an infinite
    eigenvalue.
    """

    __props__ = ()
    gufunc_signature = "(m,m),(m,m),(m,m),(m,m),(m),(m)->(m,m),(m,m)"

    def make_node(self, S, T, Q, Z, eigvals_real, eigvals_imag) -> Apply:
        inputs = list(map(pt.as_tensor, [S, T, Q, Z, eigvals_real, eigvals_imag]))
        shape = inputs[0].type.shape
        outputs = [
            pt.tensor("right_vectors", dtype="complex128", shape=shape),
            pt.tensor("left_vectors", dtype="complex128", shape=shape),
        ]
        return Apply(self, inputs, outputs)

    def infer_shape(self, node, input_shapes):  # noqa: ARG002
        return [input_shapes[0], input_shapes[0]]

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:  # noqa: ARG002
        X, Y = _schur_eigenvectors(*inputs)
        outputs[0][0] = X
        outputs[1][0] = Y

    def pullback(self, inputs, outputs, cotangents):
        raise NotImplementedError("generalized_eig supports first derivatives only.")


@register_funcify_default_op_cache_key(SchurEigenvectors)
def numba_funcify_SchurEigenvectors(op, node, **kwargs):  # noqa: ARG001
    @numba_basic.numba_njit
    def schur_eigenvectors(S, T, Q, Z, eigvals_real, eigvals_imag):
        return _schur_eigenvectors(S, T, Q, Z, eigvals_real, eigvals_imag)

    cache_version = 4
    return schur_eigenvectors, cache_version
