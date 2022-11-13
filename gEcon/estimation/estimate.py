import numpy as np
from scipy import linalg
from numba import njit
from numpy.typing import ArrayLike

# from gEcon.numba_linalg.overloads import *
from gEcon.solvers.cycle_reduction import cycle_reduction, solve_shock_matrix
from gEcon.estimation.estimation_utilities import build_system_matrices, check_finite_matrix, check_bk_condition, \
    split_random_variables, extract_prior_dict
from gEcon.estimation.kalman_filter import kalman_filter

from typing import Optional, List, Tuple


def build_and_solve(param_dict: dict,
                    sparse_datas: List,
                    vars_to_estimate: Optional[List] = None) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    A collection of functionality already in the gEcon model object, extracted for speed and memory optimizations
    when doing parallel fitting. Specifically, this function avoids the need of passing around the (potentially large)
    model object when repeatedly solving the perturbation problem for policy matrices T and R.

    Parameters
    ----------
    param_dict: dictionary of string keys float values
        A dictionary that maps parameters to be estimated to point values.
    sparse_datas: list of tuples
        A list of the equations and CSR indicies needed to construct the A, B, C, and D matrices necessary to solve for
        T and R.
    vars_to_estimate: list of strings, default: None
        The subset of variables to be estimated for. When None, the variable list is assumed to be those assigned
        priors in the GCN file. This should only be used for debugging.

    Returns
    -------
    T: array
        The "policy matrix" describing how the linear system evolves with time
    R: array
        The "selection matrix" describing how exogenous shocks enter into the linear system
    success: bool
        A flag indicating whether the system has been successfully solved. This encodes three conditions: successful
        converge of the perturbation algorithm, the size of the deterministic and stochastic norms of the
        linear solution, and the blanchard-khan conditions.

    TODO: njit this function by figuring out how to get rid of the sympy lambdify functions inside sparse_datas
    """

    res = build_system_matrices(param_dict, sparse_datas, vars_to_estimate=vars_to_estimate)
    A, B, C, D = res

    if not all([check_finite_matrix(x) for x in res]):
        T = np.zeros_like(A)
        R = np.zeros((T.shape[0], 1))
        success = False
        return T, R, success

    bk_condition_met = check_bk_condition(A, B, C, tol=1e-8)

    try:
        T, result, log_norm = cycle_reduction(A, B, C, 1000, 1e-8, False)
        R = solve_shock_matrix(B, C, D, T)
    except np.linalg.LinAlgError:
        T = np.zeros_like(A)
        R = np.zeros((T.shape[0], 1))
        success = False
        return T, R, success

    success = (result == 'Optimization successful') & (log_norm < 1e-8) & bk_condition_met

    T = np.ascontiguousarray(T)
    R = np.ascontiguousarray(R)

    return T, R, success


def build_Z_matrix(obs_variables, state_variables):
    Z = np.array([[x == var for x in state_variables] for var in obs_variables], dtype='float64')
    return Z


def build_Q_and_H(state_sigmas, shock_variables, obs_variables, obs_sigmas=None):
    k_posdef = len(shock_variables)
    k_obs = len(obs_variables)

    obs_sigmas = obs_sigmas or {}

    i = 0
    Q = np.zeros((k_posdef, k_posdef))
    for v in shock_variables:
        if v in state_sigmas.keys():
            Q[i, i] = state_sigmas[v]
        i += 1

    i = 0
    H = np.zeros((k_obs, k_obs))
    for v in obs_variables:
        if v in obs_sigmas.keys():
            H[i, i] = obs_sigmas[v]
        i += 1

    Q = np.ascontiguousarray(Q)
    H = np.ascontiguousarray(H)

    return Q, H


def evaluate_prior_logp(all_param_dict, prior_dict):
    ll = 0

    for k, v in all_param_dict.items():
        ll += prior_dict[k].logpdf(v).sum()

    return ll


def split_param_dict(all_param_dict):
    param_dict = {}
    a0_dict = {}
    P0_dict = {}

    initial_names = [x for x in all_param_dict.keys() if x.endswith('__initial')]
    initial_cov_names = [x for x in all_param_dict.keys() if x.endswith('__initial_cov')]

    for k, v in all_param_dict.items():
        if k in initial_names:
            a0_dict[k] = v
        elif k in initial_cov_names:
            P0_dict[k] = v
        else:
            param_dict[k] = v

    return param_dict, a0_dict, P0_dict


def evaluate_logp(all_param_dict, df, sparse_datas, Z, priors, shock_names, observed_vars,
                  filter_type='standard'):
    ll = evaluate_prior_logp(all_param_dict, priors)
    param_dict, a0_dict, P0_dict = split_param_dict(all_param_dict)

    if not np.isfinite(ll):
        return -np.inf, np.zeros(df.shape[0])

    param_dict, shock_dict, obs_dict = split_random_variables(param_dict, shock_names, observed_vars)
    T, R, success = build_and_solve(param_dict, sparse_datas)

    if not success:
        return -np.inf, np.zeros(df.shape[0])

    a0 = np.array(list(a0_dict.values()))[:, None] if len(a0_dict) > 0 else None
    P0 = np.eye(len(P0_dict)) * np.array(list(P0_dict.keys())) if len(P0_dict) > 0 else None

    Q, H = build_Q_and_H(shock_dict, shock_names, observed_vars, obs_dict)

    *_, ll_obs = kalman_filter(df.values, T, Z, R, H, Q, a0, P0, filter_type=filter_type)
    ll += ll_obs.sum()

    return ll, ll_obs
