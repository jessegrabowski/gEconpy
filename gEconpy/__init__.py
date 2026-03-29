import logging as _logging

from importlib.metadata import version as _version

# Side-effect imports: register JAX and Numba dispatch rules for internal
# pytensor Ops. These modules expose no public API.
import gEconpy.pytensorf.real
import gEconpy.pytensorf.real_eig  # noqa: F401

from gEconpy import classes, data, parser, plotting, solvers, utilities
from gEconpy.dynare_convert import make_mod_file
from gEconpy.model.build import model_from_gcn, statespace_from_gcn
from gEconpy.model.perfect_foresight.solve import solve_perfect_foresight
from gEconpy.model.sampling import (
    bounds_from_priors,
    sample_from_priors,
    sample_from_priors_qmc,
    sample_uniform,
    sample_uniform_from_priors,
)
from gEconpy.model.simulate import impulse_response_function, simulate
from gEconpy.model.statespace import data_from_prior, prepare_mixed_frequency_data
from gEconpy.model.statistics import (
    autocorrelation_matrix,
    autocovariance_matrix,
    build_Q_matrix,
    check_bk_condition,
    check_steady_state,
    matrix_to_dataframe,
    prior_solvability_check,
    solvability_check,
    stationary_covariance_matrix,
    summarize_perturbation_solution,
)
from gEconpy.model.steady_state import print_steady_state
from gEconpy.parser.html import print_gcn_file

_log = _logging.getLogger(__name__)
if not _log.handlers and not _logging.root.handlers:
    _log.setLevel(_logging.INFO)
    _log.addHandler(_logging.StreamHandler())

__version__ = _version("gEconpy")

__all__ = [
    "autocorrelation_matrix",
    "autocovariance_matrix",
    "bounds_from_priors",
    "build_Q_matrix",
    "check_bk_condition",
    "check_steady_state",
    "classes",
    "data",
    "data_from_prior",
    "impulse_response_function",
    "make_mod_file",
    "matrix_to_dataframe",
    "model_from_gcn",
    "parser",
    "plotting",
    "prepare_mixed_frequency_data",
    "print_gcn_file",
    "print_steady_state",
    "prior_solvability_check",
    "sample_from_priors",
    "sample_from_priors_qmc",
    "sample_uniform",
    "sample_uniform_from_priors",
    "simulate",
    "solvability_check",
    "solve_perfect_foresight",
    "solvers",
    "statespace_from_gcn",
    "stationary_covariance_matrix",
    "summarize_perturbation_solution",
    "utilities",
]
