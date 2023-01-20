from typing import Tuple

# from gEconpy.numba_linalg.overloads import *  # pylint: disable=unused-wildcard-import,wildcard-import
import numpy as np
from numba import njit
from numpy.typing import ArrayLike
from scipy import linalg

MVN_CONST = np.log(2.0 * np.pi)
EPS = 1e-12


@njit("float64[:, ::1](boolean[::1])")
def build_mask_matrix(nan_mask: ArrayLike) -> ArrayLike:
    """
    The Kalman Filter can "natively" handle missing values by treating observed states as un-observed states for
    iterations where data is not available. To do this, the Z and H matrices must be modified. This function creates
    a matrix W such that W @ Z and W @ H have zeros where data is missing.
    Parameters
    ----------
    nan_mask: array
        A 1d array of boolean flags of length n, indicating whether a state is observed in the current iteration.
    Returns
    -------
    W: array
        An n x n matrix used to mask missing values in the Z and H matrices
    """
    n = nan_mask.shape[0]
    W = np.eye(n)
    i = 0
    for flag in nan_mask:
        if flag:
            W[i, i] = 0
        i += 1

    W = np.ascontiguousarray(W)

    return W


@njit
def standard_kalman_filter(
    data: ArrayLike,
    T: ArrayLike,
    Z: ArrayLike,
    R: ArrayLike,
    H: ArrayLike,
    Q: ArrayLike,
    a0: ArrayLike,
    P0: ArrayLike,
) -> Tuple:
    """
    Parameters
    ----------
    data: array
        (T, k_observed) matrix of observed data. Data can include missing values.
    a0: array
        (k_states, 1) vector of initial states.
    P0: array
        (k_states, k_states) initial state covariance matrix
    T: array
        (k_states, k_states) transition matrix
    Z: array
        (k_states, k_observed) design matrix
    R: array
    H: array
    Q: array
    Returns
    -------
    """
    n_steps, k_obs = data.shape
    k_states, k_posdef = R.shape

    filtered_states = np.zeros((n_steps, k_states))
    predicted_states = np.zeros((n_steps + 1, k_states))
    filtered_cov = np.zeros((n_steps, k_states, k_states))
    predicted_cov = np.zeros((n_steps + 1, k_states, k_states))
    log_likelihood = np.zeros(n_steps)

    a = a0
    P = P0

    predicted_states[0] = a
    predicted_cov[0] = P

    for i in range(n_steps):
        a_filtered, a_hat, P_filtered, P_hat, ll = kalman_step(
            data[i].copy(), a, P, T, Z, R, H, Q
        )

        filtered_states[i] = a_filtered[:, 0]
        predicted_states[i + 1] = a_hat[:, 0]
        filtered_cov[i] = P_filtered
        predicted_cov[i + 1] = P_hat
        log_likelihood[i] = ll[0]

        a = a_hat
        P = P_hat

    return (
        filtered_states,
        predicted_states,
        filtered_cov,
        predicted_cov,
        log_likelihood,
    )


@njit
def kalman_step(y, a, P, T, Z, R, H, Q):
    y = y.reshape(-1, 1)
    nan_mask = np.isnan(y).ravel()
    W = build_mask_matrix(nan_mask)

    Z_masked = W @ Z
    H_masked = W @ H
    y_masked = y.copy()
    y_masked[nan_mask] = 0.0

    a_filtered, P_filtered, ll = filter_step(y_masked, Z_masked, H_masked, a, P)

    a_hat, P_hat = predict(a=a_filtered, P=P_filtered, T=T, R=R, Q=Q)

    return a_filtered, a_hat, P_filtered, P_hat, ll


@njit(
    "Tuple((float64[:, ::1], float64[:, ::1], float64[::1]))(float64[:, ::1], float64[:, ::1], float64[:, ::1], "
    "float64[:, ::1], float64[:, ::1])"
)
def filter_step(y, Z, H, a, P):
    v = y - Z @ a

    PZT = P @ Z.T
    F = Z @ PZT + H

    # Special case for if everything is missing. Abort before failing to invert F
    if np.all(Z == 0):
        a_filtered = np.atleast_2d(a).reshape((-1, 1))
        P_filtered = P
        ll = np.zeros(v.shape[0])

        return a_filtered, P_filtered, ll

    F_chol = np.linalg.cholesky(F)
    K = linalg.solve_triangular(
        F_chol, linalg.solve_triangular(F_chol, PZT.T, lower=True), trans=1, lower=True
    ).T

    I_KZ = np.eye(K.shape[0]) - K @ Z

    a_filtered = a + K @ v
    P_filtered = I_KZ @ P @ I_KZ.T + K @ H @ K.T
    P_filtered = 0.5 * (P_filtered + P_filtered.T)

    inner_term = linalg.solve_triangular(
        F_chol, linalg.solve_triangular(F_chol, v, lower=True), lower=True, trans=1
    )
    n = y.shape[0]
    ll = (
        -0.5 * (n * MVN_CONST + (v.T @ inner_term).ravel())
        - np.log(np.diag(F_chol)).sum()
    )

    return a_filtered, P_filtered, ll


@njit
def predict(a, P, T, R, Q):
    a_hat = T @ a

    P_hat = T @ P @ T.T + R @ Q @ R.T
    P_hat = 0.5 * (P_hat + P_hat.T)

    return a_hat, P_hat


@njit
def univariate_kalman_filter(
    data: ArrayLike,
    T: ArrayLike,
    Z: ArrayLike,
    R: ArrayLike,
    H: ArrayLike,
    Q: ArrayLike,
    a0: ArrayLike,
    P0: ArrayLike,
) -> Tuple:
    n_steps, k_obs = data.shape
    k_states, k_posdef = R.shape

    filtered_states = np.zeros((n_steps, k_states))
    predicted_states = np.zeros((n_steps + 1, k_states))
    filtered_cov = np.zeros((n_steps, k_states, k_states))
    predicted_cov = np.zeros((n_steps + 1, k_states, k_states))
    log_likelihood = np.zeros(n_steps)

    a = a0
    P = P0

    predicted_states[0] = a[:, 0]
    predicted_cov[0] = P

    for i in range(n_steps):
        a_filtered, a_hat, P_filtered, P_hat, ll = univariate_kalman_step(
            data[i].copy(), a, P, T, Z, R, H, Q
        )

        filtered_states[i] = a_filtered[:, 0]
        predicted_states[i + 1] = a_hat[:, 0]
        filtered_cov[i] = P_filtered
        predicted_cov[i + 1] = P_hat
        log_likelihood[i] = ll

        a = a_hat
        P = P_hat

    return (
        filtered_states,
        predicted_states,
        filtered_cov,
        predicted_cov,
        log_likelihood,
    )


@njit
def univariate_kalman_step(y, a, P, T, Z, R, H, Q):
    y = y.reshape(-1, 1)
    nan_mask = np.isnan(y).ravel()
    W = build_mask_matrix(nan_mask)

    Z_masked = W @ Z
    H_masked = W @ H
    y_masked = y.copy()
    y_masked[nan_mask] = 0.0

    a_filtered, P_filtered, ll = univariate_filter_step(
        y_masked, Z_masked, H_masked, a, P
    )

    a_hat, P_hat = predict(a=a_filtered, P=P_filtered, T=T, R=R, Q=Q)

    return a_filtered, a_hat, P_filtered, P_hat, ll


@njit
def univariate_filter_step(y_masked, Z_masked, H_masked, a, P):
    """
    Univariate step that avoids inverting the F matrix by filtering one state at a time. Good for when the H matrix
    isn't full rank (all economics problems)!
    """

    n_states = y_masked.shape[0]
    a_filtered = a.copy()
    P_filtered = P.copy()
    ll_row = np.zeros(n_states)

    for i in range(n_states):
        a_filtered, P_filtered, ll = univariate_inner_step(
            y_masked[i], Z_masked[i, :], H_masked[i, i], a_filtered, P_filtered
        )
        ll_row[i] = ll[0]

    ll = -0.5 * ((ll_row != 0).sum() * MVN_CONST + ll_row.sum())
    P_filtered = 0.5 * (P_filtered + P_filtered.T)

    return a_filtered, P_filtered, ll


@njit
def univariate_inner_step(y, Z_row, sigma_H, a, P):
    Z_row = np.atleast_2d(Z_row)
    v = y - Z_row @ a

    PZT = P @ Z_row.T
    F = Z_row @ PZT + sigma_H

    if F < EPS:
        a_filtered = a
        P_filtered = P
        ll = np.zeros(v.shape[0])
        return a_filtered, P_filtered, ll.ravel()

    K = PZT / F
    a_filtered = a + K * v
    P_filtered = P - np.outer(K, K) * F
    ll = np.log(F) + v**2 / F

    return a_filtered, P_filtered, ll.ravel()


@njit(
    "Tuple((float64[:, ::1], float64[:, ::1]))(float64[:, ::1], float64[:, ::1], float64[:, ::1], "
    "optional(float64[:, ::1]), optional(float64[:, ::1]))"
)
def make_initial_conditions(T, R, Q, a0, P0):
    if a0 is None:
        a0 = np.zeros((T.shape[0], 1))
    if P0 is None:
        P0 = linalg.solve_discrete_lyapunov(T, R @ Q @ R.T)

    return a0, P0


@njit
def kalman_filter(data, T, Z, R, H, Q, a0=None, P0=None, filter_type="standard"):
    if filter_type not in ["standard", "univariate"]:
        raise NotImplementedError(
            'Only "standard" and "univariate" kalman filters are implemented'
        )

    a0, P0 = make_initial_conditions(T, R, Q, a0, P0)

    if filter_type == "univariate":
        filter_results = univariate_kalman_filter(data, T, Z, R, H, Q, a0, P0)
    else:
        filter_results = standard_kalman_filter(data, T, Z, R, H, Q, a0, P0)

    return filter_results
