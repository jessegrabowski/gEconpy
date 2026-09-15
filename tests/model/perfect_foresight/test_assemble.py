import numpy as np
import pytest

from numpy.testing import assert_array_equal

from gEconpy.model.perfect_foresight.assemble import assemble_stacked_jacobian


def _dense_reference(period_jacobians, n_vars, n_eq, T):
    dense = np.zeros((T * n_eq, T * n_vars))
    for t, jacobian in enumerate(period_jacobians):
        rows = slice(t * n_eq, (t + 1) * n_eq)
        if t > 0:
            dense[rows, (t - 1) * n_vars : t * n_vars] = jacobian[:, :n_vars]
        dense[rows, t * n_vars : (t + 1) * n_vars] = jacobian[:, n_vars : 2 * n_vars]
        if t < T - 1:
            dense[rows, (t + 1) * n_vars : (t + 2) * n_vars] = jacobian[:, 2 * n_vars :]
    return dense


@pytest.mark.parametrize("T", [1, 2, 5])
def test_stacked_jacobian_matches_dense_block_tridiagonal(T):
    n_vars, n_eq = 3, 3
    rng = np.random.default_rng(0)
    period_jacobians = [rng.normal(size=(n_eq, 3 * n_vars)) for _ in range(T)]

    stacked = assemble_stacked_jacobian(period_jacobians, n_vars, n_eq, T)

    assert stacked.shape == (T * n_eq, T * n_vars)
    assert_array_equal(stacked.toarray(), _dense_reference(period_jacobians, n_vars, n_eq, T))


def test_entry_zero_in_one_period_keeps_its_slot_in_every_period():
    """The sparsity pattern is the union over periods, so a value that is zero at t=0 only is still placed at t=1."""
    n_vars, n_eq, T = 2, 2, 2
    period_jacobians = [np.ones((n_eq, 3 * n_vars)), np.ones((n_eq, 3 * n_vars))]
    period_jacobians[0][0, n_vars] = 0.0

    stacked = assemble_stacked_jacobian(period_jacobians, n_vars, n_eq, T)

    assert_array_equal(stacked.toarray(), _dense_reference(period_jacobians, n_vars, n_eq, T))
    assert stacked.nnz == T * n_eq * n_vars + 2 * (T - 1) * n_eq * n_vars
