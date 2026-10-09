from functools import cache
from pathlib import Path

from gEconpy import model_from_gcn, statespace_from_gcn
from gEconpy.data import get_example_gcn


@cache
def load_and_cache_model(gcn_file, infer_steady_state=True, on_unused_parameters="raise"):
    gcn_path = Path("tests") / "_resources" / "test_gcns" / gcn_file
    return model_from_gcn(
        gcn_path, verbose=False, infer_steady_state=infer_steady_state, on_unused_parameters=on_unused_parameters
    )


@cache
def load_and_cache_statespace(gcn_file):
    gcn_path = Path("tests") / "_resources" / "test_gcns" / gcn_file
    return statespace_from_gcn(gcn_path, verbose=False)


@cache
def load_and_cache_example(name: str):
    """Build one of the packaged example models, shared across every test file that renders it."""
    return model_from_gcn(get_example_gcn(name), verbose=False)
