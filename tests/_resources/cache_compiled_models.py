from functools import cache

from gEconpy import model_from_gcn, statespace_from_gcn
from gEconpy.data import get_example_gcn
from tests.conftest import TEST_GCNS


@cache
def _build_model(gcn_file: str, infer_steady_state: bool, on_unused_parameters: str):
    return model_from_gcn(
        TEST_GCNS / gcn_file,
        verbose=False,
        infer_steady_state=infer_steady_state,
        on_unused_parameters=on_unused_parameters,
    )


# The cache lives on _build_model so that the defaults are already bound by the time it sees the arguments.
# Keying on the call itself would give a caller who spells out a default an entry of its own.
def load_and_cache_model(gcn_file: str, infer_steady_state: bool = True, on_unused_parameters: str = "raise"):
    return _build_model(gcn_file, infer_steady_state, on_unused_parameters)


@cache
def load_and_cache_statespace(gcn_file: str):
    return statespace_from_gcn(TEST_GCNS / gcn_file, verbose=False)


@cache
def load_and_cache_example(name: str):
    """Build one of the packaged example models, shared across every test file that renders it."""
    return model_from_gcn(get_example_gcn(name), verbose=False)
