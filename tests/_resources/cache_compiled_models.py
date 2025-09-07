from functools import cache
from pathlib import Path

from gEconpy import model_from_gcn, statespace_from_gcn


@cache
def load_and_cache_model(gcn_file, backend, use_jax=False):
    compile_kwargs = {}
    if backend == "pytensor" and use_jax:
        compile_kwargs["mode"] = "JAX"

    gcn_path = Path("tests") / "_resources" / "test_gcns" / gcn_file
    return model_from_gcn(
        gcn_path,
        verbose=False,
        backend=backend,
        **compile_kwargs,
    )


@cache
def load_and_cache_statespace(gcn_file):
    gcn_path = Path("tests") / "_resources" / "test_gcns" / gcn_file
    return statespace_from_gcn(gcn_path, verbose=False)
