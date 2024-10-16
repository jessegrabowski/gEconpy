import os

import pytest

from gEconpy import model_from_gcn

MODEL_CACHE = {}


@pytest.fixture(scope="session")
def load_and_cache_model():
    def get_model(gcn_file, backend, force_reload=False, use_jax=False):
        global MODEL_CACHE

        if (gcn_file, backend) in MODEL_CACHE and not force_reload:
            return MODEL_CACHE[(gcn_file, backend)]

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
        MODEL_CACHE[(gcn_file, backend)] = model

        return model

    return get_model
