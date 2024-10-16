import os

from functools import cache

from gEconpy import model_from_gcn


@cache
def load_and_cache_model(gcn_file, backend, use_jax=False):
    compile_kwargs = {}
    if backend == "pytensor" and use_jax:
        compile_kwargs["mode"] = "JAX"

    gcn_path = os.path.join("tests", "Test GCNs", gcn_file)
    model = model_from_gcn(
        gcn_path,
        verbose=False,
        backend=backend,
        **compile_kwargs,
    )

    return model
