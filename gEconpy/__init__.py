import logging
import sys

from gEconpy import (
    classes,
    parser,
    plotting,
    sampling,
    shared,
    solvers,
)
from gEconpy.model.build import model_from_gcn
from gEconpy.shared import compile_to_statsmodels, make_mod_file

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler(sys.stderr)
        _log.addHandler(handler)


__version__ = "1.2.1"
__all__ = [
    "model_from_gcn",
    "classes",
    "estimation",
    "exceptions",
    "parser",
    "plotting",
    "sampling",
    "shared",
    "solvers",
    "compile_to_statsmodels",
    "make_mod_file",
]
