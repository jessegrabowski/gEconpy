import logging
import sys

from importlib.metadata import version

from gEconpy import (
    classes,
    numbaf,  #  noqa: F401
    parser,
    plotting,
    solvers,
    utilities,
)
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
from gEconpy.parser.html import print_gcn_file

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler(sys.stderr)
        _log.addHandler(handler)


__version__ = version("gEconpy")

__all__ = [
    "autocorrelation_matrix",
    "autocovariance_matrix",
    "check_bk_condition",
    "check_steady_state",
    "classes",
    "data_from_prior",
    "exceptions",
    "impulse_response_function",
    "make_mod_file",
    "matrix_to_dataframe",
    "model_from_gcn",
    "parser",
    "plotting",
    "print_gcn_file",
    "print_steady_state",
    "simulate",
    "solvers",
    "statespace_from_gcn",
    "stationary_covariance_matrix",
    "summarize_perturbation_solution",
    "utilities",
]
