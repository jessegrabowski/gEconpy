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

import logging

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)
