from functools import partial

import pytest

from gEconpy import model_from_gcn

MODEL_CACHE = {}


@pytest.fixture(scope="session")
def load_and_cache_model():
    def get_model(file_path, backend, force_reload=False, use_jax=False):
        global MODEL_CACHE

        if (file_path, backend) in MODEL_CACHE and not force_reload:
            return MODEL_CACHE[(file_path, backend)]

        compile_kwargs = {}
        if backend == "pytensor" and use_jax:
            compile_kwargs["mode"] = "JAX"

        model = model_from_gcn(
            f"tests/Test GCNs/{file_path}",
            verbose=False,
            backend=backend,
            **compile_kwargs,
        )
        MODEL_CACHE[(file_path, backend)] = model

        return model

    return get_model
