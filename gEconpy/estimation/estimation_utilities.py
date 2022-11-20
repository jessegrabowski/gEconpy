import numpy as np
import sympy as sp

from numba import njit
from scipy import linalg
from typing import List, Optional

# from gEconpy.numba_linalg.overloads import *
from gEconpy.shared.utilities import string_keys_to_sympy, sympy_keys_to_strings


@njit
def check_finite_matrix(a):
    for v in np.nditer(a):
        if not np.isfinite(v.item()):
            return False
    return True


def extract_sparse_data_from_model(model, vars_to_estimate: Optional[List] = None) -> List:
    vars_to_estimate = vars_to_estimate or model.param_priors.keys()

    param_dict = model.free_param_dict.copy()
    ss_sub_dict = string_keys_to_sympy(model.steady_state_relationships)
    calib_dict = model.calib_param_dict

    not_estimated_dict = {k: param_dict[k] for k in param_dict.keys() if k not in vars_to_estimate}

    names = ['A', 'B', 'C', 'D']
    A, B, C, D = [x.tolist() for x in model._perturbation_setup(return_F_matrices=True)]

    sparse_datas = []

    for name, matrix in zip(names, [A, B, C, D]):
        # TODO: This line causes several problems. The lambidfied functions cannot be pickled, so Windows cannot use
        #   parallel sampling with the emcee package. Also, this function itself cannot be jitted in no-python mode,
        #   which is the only step preventing the whole system from working end-to-end with numba. Needs work.

        data = tuple(
            [njit(sp.lambdify(vars_to_estimate, value.subs(ss_sub_dict).subs(calib_dict).subs(not_estimated_dict))) for row in
             matrix for (i, value) in enumerate(row) if value != 0])

        idxs = np.array([i for row in matrix for (i, value) in enumerate(row) if value != 0], dtype='int32')
        pointers = np.r_[[0], np.cumsum([sum([value != 0 for value in row]) for row in matrix])].astype('int32')
        shape = (len(matrix), len(matrix[0]))
        sparse_datas.append((data, idxs, pointers, shape))

    return sparse_datas


@njit
def matrix_from_csr_data(data, indices, idxptrs, shape):
    out = np.zeros(shape)
    for i in range(shape[0]):
        start = idxptrs[i]
        end = idxptrs[i + 1]
        s = slice(start, end)
        d_idx = range(start, end)
        col_idxs = indices[s]
        for j, d in zip(col_idxs, d_idx):
            out[i, j] = data[d]

    return out


def build_system_matrices(param_dict, sparse_datas, vars_to_estimate=None):
    result = []
    if vars_to_estimate:
        params_to_use = {k: v for k, v in param_dict.items() if k in vars_to_estimate}
    else:
        params_to_use = param_dict

    for sparse_data in sparse_datas:
        fs, indices, idxptrs, shape = sparse_data
        data = np.zeros(len(fs))
        i = 0
        for f in fs:
            data[i] = f(**params_to_use)
            i += 1
        M = matrix_from_csr_data(data, indices, idxptrs, shape)
        result.append(M)
    return result


@njit
def compute_eigenvalues(A, B, C, tol=1e-8):
    n_eq, n_vars = A.shape

    lead_var_idx = np.where(np.sum(np.abs(C), axis=0) > tol)[0]

    eqs_and_leads_idx = np.hstack((np.arange(n_vars), lead_var_idx + n_vars))

    Gamma_0 = np.vstack((np.hstack((B, C)),
                         np.hstack((-np.eye(n_eq), np.zeros((n_eq, n_eq))))))

    Gamma_1 = np.vstack((np.hstack((A, np.zeros((n_eq, n_eq)))),
                         np.hstack((np.zeros((n_eq, n_eq)), np.eye(n_eq)))))
    Gamma_0 = Gamma_0[eqs_and_leads_idx, :][:, eqs_and_leads_idx]
    Gamma_1 = Gamma_1[eqs_and_leads_idx, :][:, eqs_and_leads_idx]

    A, B, alpha, beta, Q, Z = linalg.ordqz(-Gamma_0, Gamma_1, sort='ouc', output='complex')

    gev = np.vstack((np.diag(A), np.diag(B))).T

    eigenval = gev[:, 1] / (gev[:, 0] + tol)
    pos_idx = np.where(np.abs(eigenval) > 0)
    eig = np.zeros(((np.abs(eigenval) > 0).sum(), 3))
    eig[:, 0] = np.abs(eigenval)[pos_idx]
    eig[:, 1] = np.real(eigenval)[pos_idx]
    eig[:, 2] = np.imag(eigenval)[pos_idx]

    sorted_idx = np.argsort(eig[:, 0])

    return eig[sorted_idx, :]


@njit
def check_bk_condition(A, B, C, tol=1e-8):
    n_forward = int((C.sum(axis=0) > 0).sum())

    try:
        eig = compute_eigenvalues(A, B, C, tol)
    # TODO: ValueError is the correct exception to raise here, but numba complains
    except Exception:
        return False

    n_g_one = (eig[:, 0] > 1).sum()
    return n_forward <= n_g_one


def split_random_variables(param_dict, shock_names, obs_names):
    out_param_dict = {}
    shock_dict = {}
    obs_dict = {}

    for k, v in param_dict.items():
        if k in shock_names:
            shock_dict[k] = v
        elif k in obs_names:
            obs_dict[k] = v
        else:
            out_param_dict[k] = v

    return out_param_dict, shock_dict, obs_dict


def extract_prior_dict(model):
    prior_dict = {}

    prior_dict.update(model.param_priors)
    prior_dict.update({k:model.shock_priors[k].rv_params['scale'] for k in model.shock_priors.keys()})
    prior_dict.update(model.observation_noise_priors)

    return prior_dict
