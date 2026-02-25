from functools import cache
from pathlib import Path

from gEconpy import model_from_gcn, statespace_from_gcn


@cache
def load_and_cache_model(gcn_file, infer_steady_state=True):
    mode = "FAST_RUN"
    gcn_path = Path("tests") / "_resources" / "test_gcns" / gcn_file
    return model_from_gcn(gcn_path, verbose=False, infer_steady_state=infer_steady_state, mode=mode)


@cache
def load_and_cache_statespace(gcn_file):
    gcn_path = Path("tests") / "_resources" / "test_gcns" / gcn_file
    return statespace_from_gcn(gcn_path, verbose=False)
