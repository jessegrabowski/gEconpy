import logging
import sys

from gEconpy import (
    classes,
    parser,
    plotting,
    shared,
    solvers,
)
from gEconpy.model.build import model_from_gcn, statespace_from_gcn
from gEconpy.model.model import (
    autocorrelation_matrix,
    autocovariance_matrix,
    bk_condition,
    impulse_response_function,
    simulate,
    stationary_covariance_matrix,
    summarize_perturbation_solution,
)
from gEconpy.shared import make_mod_file

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler(sys.stderr)
        _log.addHandler(handler)


__version__ = "2.0.0"
__all__ = [
    "model_from_gcn",
    "statespace_from_gcn",
    "simulate",
    "impulse_response_function",
    "summarize_perturbation_solution",
    "stationary_covariance_matrix",
    "autocovariance_matrix",
    "autocorrelation_matrix",
    "bk_condition",
    "classes",
    "exceptions",
    "parser",
    "plotting",
    "shared",
    "solvers",
    "make_mod_file",
]
