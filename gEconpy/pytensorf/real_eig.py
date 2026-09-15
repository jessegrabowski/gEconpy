import numpy as np
import pytensor.tensor as pt

from pytensor.gradient import DisconnectedType
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import TensorLike, TensorVariable
from pytensor.tensor.blockwise import Blockwise


class RealEig(Op):
    """Op computing eigenvalues of a real matrix as two real tensors, one per component.

    Wraps ``numpy.linalg.eig``, splits the eigenvalues into real and imaginary parts, and sorts them by
    ascending modulus. Only first-order derivatives are supported.
    """

    __props__ = ()
    gufunc_signature = "(m,m)->(m),(m)"

    def make_node(self, M):
        M = pt.as_tensor_variable(M)
        if M.ndim != 2:
            raise ValueError(
                f"RealEig requires a square matrix, but got an input with ndim={M.ndim}. Pass a 2-d tensor, or wrap "
                "the Op in Blockwise for batched input."
            )
        n = M.type.shape[0]
        outputs = [pt.vector(dtype=M.dtype, shape=(n,)), pt.vector(dtype=M.dtype, shape=(n,))]
        return Apply(self, [M], outputs)

    def perform(self, _node, inputs, outputs):
        (M,) = inputs
        eigvals = np.linalg.eig(M)[0]
        idx = np.argsort(np.abs(eigvals))
        eigvals = eigvals[idx]
        outputs[0][0] = np.real(eigvals).astype(M.dtype)
        outputs[1][0] = np.imag(eigvals).astype(M.dtype)

    def pullback(self, inputs, outputs, cotangents):
        (M,) = inputs
        g_real, g_imag = cotangents

        if isinstance(g_real.type, DisconnectedType):
            g_real = pt.zeros_like(outputs[0])
        if isinstance(g_imag.type, DisconnectedType):
            g_imag = pt.zeros_like(outputs[1])

        # Recompute the eigenvectors on the graph, the same strategy as JAX's eigvals VJP, and sort them to match
        # the modulus-ascending order that perform uses.
        eigvals, V = pt.linalg.eig(M)
        V = V[:, pt.argsort(pt.abs(eigvals))]

        # M_bar = Re(V^{-T} diag(g) V^T) with the complex cotangent g = g_real - i * g_imag.
        g = g_real.astype("complex128") - 1j * g_imag.astype("complex128")
        V_inv = pt.linalg.solve(V, pt.eye(M.shape[0], dtype="complex128"))
        M_bar = V_inv.T @ pt.diag(g) @ V.T

        return [M_bar.real]


def real_eig(M: TensorLike) -> tuple[TensorVariable, TensorVariable]:
    """Compute eigenvalues of a real matrix, returning real and imaginary parts separately.

    The outputs are real-valued tensors, so reverse-mode differentiation through both components works, which
    :func:`pytensor.tensor.linalg.eig` does not support. Eigenvalues are sorted by ascending modulus.

    Parameters
    ----------
    M : TensorVariable
        A real-valued square matrix of shape ``(n, n)``, or anything :func:`pytensor.tensor.as_tensor_variable`
        accepts. Leading batch dimensions are supported through ``Blockwise``.

    Returns
    -------
    eigvals_real : TensorVariable
        Real parts of the eigenvalues, shape ``(n,)``.
    eigvals_imag : TensorVariable
        Imaginary parts of the eigenvalues, shape ``(n,)``.

    Examples
    --------
    Differentiate the sum of the real parts with respect to the matrix, which ``eig`` refuses because its output is
    complex:

    .. code-block:: python

        import pytensor.tensor as pt
        from gEconpy.pytensorf.real_eig import real_eig

        M = pt.dmatrix("M")
        re, im = real_eig(M)
        modulus = pt.sqrt(re**2 + im**2)
        grad_re = pt.grad(re.sum(), M)
    """
    M = pt.as_tensor_variable(M)
    return Blockwise(RealEig())(M)


try:
    import jax.numpy as jnp

    from pytensor.link.jax.dispatch.basic import jax_funcify

    @jax_funcify.register(RealEig)
    def jax_funcify_RealEig(op, node, **kwargs):  # noqa: ARG001
        def real_eig_jax(M):
            eigvals = jnp.linalg.eigvals(M)
            idx = jnp.argsort(jnp.abs(eigvals))
            eigvals = eigvals[idx]
            return eigvals.real, eigvals.imag

        return real_eig_jax


except ImportError:
    pass


try:
    from pytensor.link.numba.dispatch.basic import (
        numba_njit,
        register_funcify_default_op_cache_key,
    )

    @register_funcify_default_op_cache_key(RealEig)
    def numba_funcify_RealEig(op, node, **kwargs):  # noqa: ARG001
        @numba_njit
        def real_eig_numba(M):
            M_c = M.astype(np.complex128)
            eigvals = np.linalg.eig(M_c)[0]
            idx = np.argsort(np.abs(eigvals))
            eigvals = eigvals[idx]
            return np.real(eigvals).astype(M.dtype), np.imag(eigvals).astype(M.dtype)

        cache_version = 1
        return real_eig_numba, cache_version

except ImportError:
    pass
