import logging
import sys

from gEconpy import (
    classes,
    numbaf,  #  noqa: F401
    parser,
    plotting,
    solvers,
    utilities,
)
from gEconpy._version import get_versions
from gEconpy.dynare_convert import make_mod_file
from gEconpy.model.build import model_from_gcn, statespace_from_gcn
from gEconpy.model.model import (
    autocorrelation_matrix,
    autocovariance_matrix,
    check_bk_condition,
    check_steady_state,
    impulse_response_function,
    matrix_to_dataframe,
    simulate,
    stationary_covariance_matrix,
    summarize_perturbation_solution,
)
from gEconpy.model.statespace import data_from_prior
from gEconpy.model.steady_state import print_steady_state

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler(sys.stderr)
        _log.addHandler(handler)


__version__ = get_versions()["version"]

__all__ = [
    "model_from_gcn",
    "statespace_from_gcn",
    "simulate",
    "impulse_response_function",
    "summarize_perturbation_solution",
    "stationary_covariance_matrix",
    "autocovariance_matrix",
    "autocorrelation_matrix",
    "check_bk_condition",
    "check_steady_state",
    "matrix_to_dataframe",
    "print_steady_state",
    "classes",
    "exceptions",
    "parser",
    "plotting",
    "utilities",
    "solvers",
    "make_mod_file",
    "data_from_prior",
]
