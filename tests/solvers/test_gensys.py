import numpy as np
import pytest

from numpy.testing import assert_allclose

from gEconpy.model.build import model_from_gcn
from gEconpy.solvers.gensys import (
    build_u_v_d,
    determine_n_unstable,
    interpret_gensys_output,
    solve_policy_function_with_gensys,
    split_matrix_on_eigen_stability,
)
from tests._resources.cache_compiled_models import load_and_cache_model

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
    div, n_unstable, zxz = determine_n_unstable(ALPHA, BETA, div=1.01, realsmall=1e-6)

    assert div == 1.01
    assert n_unstable == 5
    assert zxz is False


def test_determine_n_unstable_flags_coincident_zeros():
    """A trailing (alpha, beta) pair below realsmall signals a singular pencil, whatever came before it."""
    alpha = np.array([2.0 + 0.0j, 1e-9 + 0.0j])
    beta = np.array([1.0 + 0.0j, 1e-9 + 0.0j])

    _, n_unstable, zxz = determine_n_unstable(alpha, beta, div=1.01, realsmall=1e-6)

    assert n_unstable == 0
    assert zxz is True


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


@pytest.mark.parametrize(
    ("eta", "invalid_system"),
    [(np.ones((3, 2), dtype=np.complex128), True), (np.zeros((3, 0), dtype=np.complex128), False)],
    ids=["invalid_system", "empty_eta"],
)
def test_build_u_v_d_returns_empty_components(eta, invalid_system):
    u_eta, v_eta, d_eta, big_ev = build_u_v_d(eta, invalid_system=invalid_system)

    assert u_eta.shape == (3, 0)
    assert v_eta.shape == (eta.shape[1], 0)
    assert d_eta.shape == (0,)
    assert big_ev.shape == (0,)


@pytest.mark.parametrize(
    ("eu", "expected"),
    [
        ([-2, -2, 0], "Coincident zeros"),
        ([-1, 0, 3], "indeterminate. There are 3 loose"),
        ([1, -1, 0], "not unique (sunspots)"),
        ([0, 0, 0], "Solution does not exist"),
        ([1, 0, 2], "Solution exists, but is not unique."),
        ([1, 1, 0], "unique solution"),
        ([5, 5, 0], "Unknown return code"),
    ],
)
def test_interpret_gensys_output(eu, expected):
    message = interpret_gensys_output(eu)
    assert message.startswith("Gensys return codes: " + " ".join(map(str, eu)))
    assert expected in message


def test_policy_only_return_keeps_existence_codes():
    mod = load_and_cache_model("one_block_1_ss.gcn")
    A, B, C, D = mod.linearize_model(verbose=False, steady_state_kwargs={"verbose": False, "progressbar": False})
    n = A.shape[0]

    G_1, eu = solve_policy_function_with_gensys(A, B, C, D, return_all_matrices=False)
    *_, eu_full, _ = solve_policy_function_with_gensys(A, B, C, D)
    T = G_1[:n, :n]

    assert eu == eu_full == [1, 1, 0]
    assert_allclose(A + B @ T + C @ T @ T, 0.0, atol=1e-8)


def test_mistimed_model_reports_non_unique_solution():
    mod = model_from_gcn("tests/_resources/test_gcns/pert_fails.gcn", verbose=False, on_unused_parameters="ignore")
    A, B, C, D = mod.linearize_model(verbose=False, steady_state_kwargs={"verbose": False, "progressbar": False})

    _, eu = solve_policy_function_with_gensys(A, B, C, D, return_all_matrices=False)

    assert eu == [1, 0, 2]
