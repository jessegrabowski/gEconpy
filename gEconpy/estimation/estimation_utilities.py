from typing import Callable, Dict, List, Optional, Tuple

import numba as nb  # pylint: disable=unused-import
import numpy as np
import sympy as sp
from numba import njit
from scipy import linalg
from sympy import Expr

# from gEconpy.numba_linalg.overloads import *
from gEconpy.shared.utilities import string_keys_to_sympy


def convert_to_numba_function(expr: Expr, vars: List[str]) -> Callable:
    """
    Convert a sympy expression into a Numba-compiled function.

    Parameters
    ----------
    expr : sympy.Expr
        The sympy expression to be converted.
    vars : List[str]
        A list of strings containing the names of the variables in the expression.

    Returns
    -------
    numba.types.function
        A Numba-compiled function equivalent to the input expression.

    Notes
    -----
    The function returned by this function is pickleable.
    """
    code = sp.printing.ccode(expr)
    # The code string will contain a single line, so we add line breaks to make it a valid block of code
    code = "@nb.njit\ndef f({}):\n{}\n    return {}".format(
        ",".join(vars), " " * 4, code
    )
    # Compile the code and return the resulting function
    exec(code)
    return locals()["f"]


@njit
def check_finite_matrix(a):
    for v in np.nditer(a):
        if not np.isfinite(v.item()):
            return False
    return True


def extract_sparse_data_from_model(
    model, vars_to_estimate: Optional[List] = None
) -> List:
    """
    Extract sparse data from a DSGE model.

    Parameters
    ----------
    model : object
        A gEconpy model object.
    vars_to_estimate : list, optional
        A list of variables to estimate. The default is None, which estimates all variables.

    Returns
    -------
    list
        A list of sparse data.
    """

    vars_to_estimate = vars_to_estimate or model.param_priors.keys()

    param_dict = model.free_param_dict.copy()
    ss_sub_dict = string_keys_to_sympy(model.steady_state_relationships)
    calib_dict = model.calib_param_dict

    not_estimated_dict = {
        k: param_dict[k] for k in param_dict.keys() if k not in vars_to_estimate
    }

    names = ["A", "B", "C", "D"]
    A, B, C, D = (x.tolist() for x in model._perturbation_setup(return_F_matrices=True))

    sparse_datas = []

    for name, matrix in zip(names, [A, B, C, D]):
        data = []
        idxs = []
        pointers = [0]

        for row in matrix:
            for i, value in enumerate(row):
                if value != 0:
                    # Convert the sympy expression into a Numba-compatible function using the above function
                    numba_func = convert_to_numba_function(
                        value.subs(ss_sub_dict)
                        .subs(calib_dict)
                        .subs(not_estimated_dict),
                        vars_to_estimate,
                    )
                    data.append(numba_func)
                    idxs.append(i)
            pointers.append(len(idxs))

        shape = (len(matrix), len(matrix[0]))
        sparse_datas.append((data, idxs, pointers, shape))

    return sparse_datas


@njit
def matrix_from_csr_data(
    data: np.ndarray, indices: np.ndarray, idxptrs: np.ndarray, shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert a CSR matrix into a dense numpy array.

    Parameters
    ----------
    data : np.ndarray
        The data stored in the CSR matrix.
    indices : np.ndarray
        The column indices for the non-zero values in `data`.
    idxptrs : np.ndarray
        The index pointers for the CSR matrix.
    shape : tuple[int, int]
        The shape of the dense matrix to create.

    Returns
    -------
    np.ndarray
        The dense matrix representation of the CSR matrix.
    """
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


def build_system_matrices(
    param_dict: Dict[str, float],
    sparse_datas: List[Tuple[Callable, np.ndarray, np.ndarray, Tuple[int, int]]],
    vars_to_estimate: Optional[List[str]] = None,
) -> List[np.ndarray]:
    """
    Build system matrices for a DSGE model.

    This function builds the A, B, C, and D matrices for a DSGE model given a set of parameters
    and pre-computed sparse data.

    Parameters
    ----------
    param_dict : dict
        Dictionary of parameter values
    sparse_datas : list of tuples
        List of tuples, each tuple representing sparse data for a single matrix. The tuple contains the following
        elements:
        data : numpy array
            Array of values to be placed in the matrix
        indices : numpy array
            Array of column indices for the non-zero values in the matrix
        idxptrs : numpy array
            Array of row pointers for the non-zero values in the matrix
        shape : tuple
            Shape of the matrix as a tuple (n_rows, n_cols)
    vars_to_estimate : list of str, optional
        List of parameter names to use in building the matrices, by default None
    Returns
    -------
    list of numpy arrays
        List of matrices A, B, C, and D
    """

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
    """
    Given the log-linearized coefficient matrices A, B, and C at times t-1, t, and t+1 respectively, compute the
    eigenvalues of the DSGE system. These eigenvalues are used to determine stability of the DSGE system.

    Parameters
    ----------
    A : np.ndarray
        The log-linearized coefficient matrix of the DSGE system at time t-1
    B : np.ndarray
        The log-linearized coefficient matrix of the DSGE system at time t
    C : np.ndarray
        The log-linearized coefficient matrix of the DSGE system at time t+1
    tol : float, optional
        The tolerance used to check for stability, by default 1e-8

    Returns
    -------
    np.ndarray
        The eigenvalues of the DSGE system, sorted by the magnitude of the real part. Each row of the output array
        contains the magnitude, real part, and imaginary part of an eigenvalue.
    """

    n_eq, n_vars = A.shape

    lead_var_idx = np.where(np.sum(np.abs(C), axis=0) > tol)[0]

    eqs_and_leads_idx = np.hstack((np.arange(n_vars), lead_var_idx + n_vars))

    Gamma_0 = np.vstack(
        (np.hstack((B, C)), np.hstack((-np.eye(n_eq), np.zeros((n_eq, n_eq)))))
    )

    Gamma_1 = np.vstack(
        (
            np.hstack((A, np.zeros((n_eq, n_eq)))),
            np.hstack((np.zeros((n_eq, n_eq)), np.eye(n_eq))),
        )
    )
    Gamma_0 = Gamma_0[eqs_and_leads_idx, :][:, eqs_and_leads_idx]
    Gamma_1 = Gamma_1[eqs_and_leads_idx, :][:, eqs_and_leads_idx]

    A, B, alpha, beta, Q, Z = linalg.ordqz(
        -Gamma_0, Gamma_1, sort="ouc", output="complex"
    )

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
    """
    Check the Blanchard-Kahn condition for the DSGE model specified by the log linearized coefficient matrices
    A (t-1), B (t), and C (t+1).

    This function computes the eigenvalues of the DSGE system and checks if the number of forward-looking variables
    is less than or equal to the number of eigenvalues greater than 1. The Blanchard-Kahn condition ensures the
    stability of the rational expectations equilibrium of the model.

    Parameters
    ----------
    A : numpy.ndarray
        The log-linearized coefficient matrix at time t-1
    B : numpy.ndarray
        The log-linearized coefficient matrix at time t
    C : numpy.ndarray
        The log-linearized coefficient matrix at time t+1
    tol : float, optional
        The tolerance for eigenvalues that are considered equal to 1, by default 1e-8

    Returns
    -------
    bool
        True if the Blanchard-Kahn condition is satisfied, else False

    References
    ----------
    Blanchard, Olivier Jean, and Charles M. Kahn. "The solution of linear difference models under rational
    expectations." Econometrica: Journal of the Econometric Society (1980): 1305-1311.
    """

    n_forward = int((C.sum(axis=0) > 0).sum())

    try:
        eig = compute_eigenvalues(A, B, C, tol)
    # TODO: ValueError is the correct exception to raise here, but numba complains
    except Exception:
        return False

    n_g_one = (eig[:, 0] > 1).sum()
    return n_forward <= n_g_one


def split_random_variables(param_dict, shock_names, obs_names):
    """
    Split a dictionary of parameters into dictionaries of shocks, observables, and other variables.

    Parameters
    ----------
    param_dict : Dict[str, float]
        A dictionary of parameters and their values.
    shock_names : List[str]
        A list of the names of shock variables.
    obs_names : List[str]
        A list of the names of observable variables.

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]
        A tuple containing three dictionaries: the first has parameters, the second has
        all shock variances parameters, and the third has observation noise variances.
    """

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
    """
    Extract the prior distributions from a gEconModel object.

    Parameters
    ----------
    model : gEconModel
        The gEconModel object to extract priors from.

    Returns
    -------
    prior_dict : dict
        A dictionary containing the prior distributions for the model's parameters, shocks, and observation noise.
    """
    prior_dict = {}

    prior_dict.update(model.param_priors)
    prior_dict.update(
        {k: model.shock_priors[k].rv_params["scale"] for k in model.shock_priors.keys()}
    )
    prior_dict.update(model.observation_noise_priors)

    return prior_dict
