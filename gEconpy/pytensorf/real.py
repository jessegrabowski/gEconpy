"""JAX dispatch for the pytensor scalar Real Op."""

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
