import numpy as np
from numba import njit

# from gEconpy.numba_linalg.overloads import *


@njit
def predict(a, P, T, R, Q):
    a_hat = T @ a

    P_hat = T @ P @ T.T + R @ Q @ R.T
    P_hat = 0.5 * (P_hat + P_hat.T)

    return a_hat, P_hat


@njit
def kalman_smoother(T, R, Q, filtered_states, filtered_covariances):
    n_steps, k_states = filtered_states.shape

    smoothed_states = np.zeros((n_steps, k_states))
    smoothed_covariances = np.zeros((n_steps, k_states, k_states))

    a_smooth = filtered_states[-1].copy()
    P_smooth = filtered_covariances[-1].copy()

    smoothed_states[-1] = a_smooth
    smoothed_covariances[-1] = P_smooth

    for t in range(n_steps - 1, -1, -1):
        a = filtered_states[t]
        P = filtered_covariances[t]
        a_smooth, P_smooth = smoother_step(a, P, a_smooth, P_smooth, T, R, Q)

        smoothed_states[t] = a_smooth
        smoothed_covariances[t] = P_smooth

    return smoothed_states, smoothed_covariances


@njit
def smoother_step(a, P, a_smooth, P_smooth, T, R, Q):
    a_hat, P_hat = predict(a, P, T, R, Q)

    # Use pinv, otherwise P_hat is singular when there is missing data
    smoother_gain = (np.linalg.pinv(P_hat) @ T @ P).T

    a_smooth_next = a + smoother_gain @ (a_smooth - a_hat)
    P_smooth_next = P + smoother_gain @ (P_smooth - P_hat) @ smoother_gain.T

    return a_smooth_next, P_smooth_next
