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
@pytest.mark.parametrize(("n_eq", "n_vars"), [(3, 3), (2, 4), (4, 2)], ids=["square", "wide", "tall"])
def test_stacked_jacobian_matches_dense_block_tridiagonal(T, n_eq, n_vars):
    rng = np.random.default_rng(0)
    period_jacobians = [rng.normal(size=(n_eq, 3 * n_vars)) for _ in range(T)]
    dense_pattern = np.ones((n_eq, 3 * n_vars), dtype=bool)

    stacked = assemble_stacked_jacobian(period_jacobians, dense_pattern, n_vars, n_eq, T)

    assert stacked.shape == (T * n_eq, T * n_vars)
    assert_array_equal(stacked.toarray(), _dense_reference(period_jacobians, n_vars, n_eq, T))


def test_structurally_nonzero_entry_keeps_its_slot_when_zero_in_every_period():
    """A structural entry that vanishes at the current iterate must stay in the pattern, in every period."""
    n_vars, n_eq, T = 2, 2, 2
    period_jacobians = np.ones((T, n_eq, 3 * n_vars))
    period_jacobians[:, 0, n_vars] = 0.0
    dense_pattern = np.ones((n_eq, 3 * n_vars), dtype=bool)

    stacked = assemble_stacked_jacobian(period_jacobians, dense_pattern, n_vars, n_eq, T)

    coo = stacked.tocoo()
    stored = set(zip(coo.row.tolist(), coo.col.tolist(), strict=True))
    assert all((t * n_eq, t * n_vars) in stored for t in range(T))

    assert_array_equal(stacked.toarray(), _dense_reference(period_jacobians, n_vars, n_eq, T))
    assert stacked.nnz == T * n_eq * n_vars + 2 * (T - 1) * n_eq * n_vars


def test_structurally_zero_entries_are_left_out():
    n_vars, n_eq, T = 2, 2, 3
    rng = np.random.default_rng(0)
    period_jacobians = [rng.normal(size=(n_eq, 3 * n_vars)) for _ in range(T)]
    pattern = np.ones((n_eq, 3 * n_vars), dtype=bool)
    pattern[1, n_vars] = False

    stacked = assemble_stacked_jacobian(period_jacobians, pattern, n_vars, n_eq, T)

    periods = np.arange(T)
    expected = _dense_reference(period_jacobians, n_vars, n_eq, T)
    expected[periods * n_eq + 1, periods * n_vars] = 0.0
    assert_array_equal(stacked.toarray(), expected)
    assert stacked.nnz == T * (n_eq * n_vars - 1) + 2 * (T - 1) * n_eq * n_vars


def test_wrong_pattern_shape_is_rejected():
    n_vars, n_eq, T = 2, 2, 1

    with pytest.raises(ValueError, match="sparsity_pattern has shape"):
        assemble_stacked_jacobian([np.ones((n_eq, 3 * n_vars))], np.ones((n_eq, n_vars), dtype=bool), n_vars, n_eq, T)


def test_pattern_with_an_empty_block_assembles():
    """A purely backward-looking model has no ``y_{t+1}`` dependence at all, leaving that block with no entries."""
    n_vars, n_eq, T = 2, 2, 3
    rng = np.random.default_rng(0)
    period_jacobians = [rng.normal(size=(n_eq, 3 * n_vars)) for _ in range(T)]
    pattern = np.ones((n_eq, 3 * n_vars), dtype=bool)
    pattern[:, 2 * n_vars :] = False

    stacked = assemble_stacked_jacobian(period_jacobians, pattern, n_vars, n_eq, T)

    expected = _dense_reference(period_jacobians, n_vars, n_eq, T)
    for t in range(T - 1):
        expected[t * n_eq : (t + 1) * n_eq, (t + 1) * n_vars : (t + 2) * n_vars] = 0.0
    assert_array_equal(stacked.toarray(), expected)
    assert stacked.nnz == T * n_eq * n_vars + (T - 1) * n_eq * n_vars
