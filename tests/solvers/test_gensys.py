import numpy as np

from numpy.testing import assert_allclose

from gEconpy.solvers.gensys import (
    build_u_v_d,
    determine_n_unstable,
    split_matrix_on_eigen_stability,
)

# Diagonals of the upper-triangular QZ factors of two seed=1337 normal matrices, taken from MATLAB's qz.
ALPHA = np.array(
    [
        -2.0123 - 0.5490j,
        -1.7594 + 0.4800j,
        0.9347 - 0.1598j,
        0.9237 + 0.1579j,
        1.0847 + 0.0000j,
    ]
)
BETA = np.array(
    [
        2.2056 + 0.0000j,
        1.9284 + 0.0000j,
        2.4670 + 0.0000j,
        2.4382 + 0.0000j,
        1.3904 + 0.0000j,
    ]
)


def test_determine_n_unstable():
    div, n_unstable, zxz = determine_n_unstable(ALPHA, BETA, 1.01, realsmall=1e-6)

    assert div == 1.01
    assert n_unstable == 5
    assert zxz is False


def test_determine_n_unstable_infers_div():
    # Starting from div = 1.01, the first (alpha, beta) ratio sits in (1 + eps, 1.01], so the Sims heuristic halves
    # toward 1: div := (1 + 1.005) / 2 = 1.0025. The check |beta| > div * |alpha| fires on the same iteration with
    # the shrunk div, so both eigenvalues are classified as unstable.
    alpha = np.array([1.0 + 0.0j, 1.0 + 0.0j])
    beta = np.array([1.005 + 0.0j, 2.0 + 0.0j])

    div, n_unstable, zxz = determine_n_unstable(alpha, beta, div=None, realsmall=1e-6)

    assert_allclose(div, 0.5 * (1 + 1.005))
    assert n_unstable == 2
    assert zxz is False


def test_split_matrix_on_eigen_stability():
    Q = np.arange(25, dtype=float).reshape(5, 5)
    n_unstable = 3

    Q1, Q2 = split_matrix_on_eigen_stability(Q, n_unstable)

    assert_allclose(Q1, Q[:2, :])
    assert_allclose(Q2, Q[2:, :])


def test_build_u_v_d_keeps_only_nonzero_singular_values():
    rng = np.random.default_rng(1337)
    rank_two = rng.normal(size=(5, 2)) @ rng.normal(size=(2, 4))
    eta = rank_two.astype(np.complex128)

    u_eta, v_eta, d_eta, big_ev = build_u_v_d(eta, realsmall=1e-10)

    assert d_eta.shape == (2,)
    assert big_ev.tolist() == [0, 1]
    assert_allclose(u_eta @ np.diag(d_eta) @ v_eta.conj().T, eta, atol=1e-12)


def test_build_u_v_d_invalid_system_returns_empty_components():
    eta = np.ones((3, 2), dtype=np.complex128)

    u_eta, v_eta, d_eta, big_ev = build_u_v_d(eta, invalid_system=True)

    assert u_eta.shape == (3, 0)
    assert v_eta.shape == (2, 0)
    assert d_eta.shape == (0,)
    assert big_ev.shape == (0,)
