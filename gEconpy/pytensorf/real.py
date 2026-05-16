"""JAX and numba dispatches for the pytensor scalar Real Op.

Pytensor's built-in numba dispatch for `ScalarOp` inspects `op.nfunc_spec` to
find a numpy-named implementation. `Real` doesn't set `nfunc_spec`, so without
this registration it falls back to object mode and raises a `UserWarning`.
"""

try:
    import jax.numpy as jnp

    from pytensor.link.jax.dispatch import jax_funcify
    from pytensor.scalar import Real

    @jax_funcify.register(Real)
    def jax_funcify_Real(op, node, **kwargs):  # noqa: ARG001
        def real_jax(x):
            return jnp.real(x)

        return real_jax

except ImportError:
    pass


try:
    import numpy as np

    from pytensor.link.numba.dispatch import basic as numba_basic
    from pytensor.link.numba.dispatch.basic import register_funcify_and_cache_key
    from pytensor.link.numba.dispatch.scalar import scalar_op_cache_key
    from pytensor.scalar import Real

    @register_funcify_and_cache_key(Real)
    def numba_funcify_Real(op, node, **kwargs):  # noqa: ARG001
        @numba_basic.numba_njit
        def real(x):
            return np.real(x)

        return real, scalar_op_cache_key(op)

except ImportError:
    pass
