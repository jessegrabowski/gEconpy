import numpy as np
import pytensor.tensor as pt

from pytensor.gradient import DisconnectedType
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.blockwise import Blockwise


class RealEig(Op):
    """Eigenvalue decomposition returning real and imaginary parts as separate real tensors.

    Wraps ``numpy.linalg.eig``, splits the result into two real-valued outputs, and
    sorts by ascending modulus.  The VJP uses the standard eigenvalue perturbation
    formula (arXiv:1701.00392 eq 4.77):
    ``M_bar = Re(V^{-T} diag(g_alpha - i * g_beta) V^T)``.
    Only first-order derivatives are supported (see jax-ml/jax#2748).
    """

    __props__ = ()
    gufunc_signature = "(m,m)->(m),(m)"

    def make_node(self, M):
        M = pt.as_tensor_variable(M)
        out_shape = M.type.shape[0]
        if M.ndim != 2:
            raise ValueError(f"RealEig requires a 2-d matrix, got ndim={M.ndim}")
        outputs = [pt.vector(dtype=M.dtype, shape=(out_shape,)), pt.vector(dtype=M.dtype, shape=(out_shape,))]
        return Apply(self, [M], outputs)

    def perform(self, _node, inputs, outputs):
        (M,) = inputs
        eigvals = np.linalg.eig(M)[0]
        idx = np.argsort(np.abs(eigvals))
        eigvals = eigvals[idx]
        outputs[0][0] = np.real(eigvals).astype(M.dtype)
        outputs[1][0] = np.imag(eigvals).astype(M.dtype)

    def L_op(self, inputs, outputs, output_grads):
        (M,) = inputs
        g_real, g_imag = output_grads

        if isinstance(g_real.type, DisconnectedType):
            g_real = pt.zeros_like(outputs[0])
        if isinstance(g_imag.type, DisconnectedType):
            g_imag = pt.zeros_like(outputs[1])

        # Recompute eigenvectors — same strategy as JAX's eigvals VJP.
        _eigvals, V = pt.linalg.eig(M)

        # Sort to match our modulus-ascending ordering in perform.
        sort_idx = pt.argsort(pt.abs(_eigvals))
        V = V[:, sort_idx]

        # Complex gradient vector: g = g_bar_real - i * g_bar_imag
        g = g_real.astype("complex128") - 1j * g_imag.astype("complex128")

        # VJP: M̄ = Re(V⁻ᵀ diag(g) Vᵀ)
        V_inv = pt.linalg.solve(V, pt.eye(M.shape[0], dtype="complex128"))
        M_bar = V_inv.T @ pt.diag(g) @ V.T

        return [M_bar.real]


def real_eig(M):
    r"""Compute eigenvalues of a real matrix, returning real and imaginary parts separately.

    Unlike ``pt.linalg.eig``, the outputs are real-valued tensors, so reverse-mode
    differentiation through both components works out of the box.  Eigenvalues are
    sorted by ascending modulus.

    Parameters
    ----------
    M : array_like or TensorVariable
        A real-valued square matrix of shape ``(n, n)``.

    Returns
    -------
    eigvals_real : TensorVariable
        Real parts of the eigenvalues, shape ``(n,)``.
    eigvals_imag : TensorVariable
        Imaginary parts of the eigenvalues, shape ``(n,)``.

    Examples
    --------
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
