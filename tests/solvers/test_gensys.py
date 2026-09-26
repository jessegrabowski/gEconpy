import numpy as np
import pytest

from numpy.testing import assert_allclose

from gEconpy.model.build import model_from_gcn
from gEconpy.solvers.gensys import (
    _thin_svd_and_rank,
    interpret_gensys_output,
    solve_policy_function_with_gensys,
    split_matrix_on_eigen_stability,
)
from tests._resources.cache_compiled_models import load_and_cache_model


# Diagonals of the upper-triangular QZ factors of two seed=1337 normal matrices, taken from MATLAB's qz.
def test_thin_svd_and_rank_keeps_only_nonzero_singular_values():
    rng = np.random.default_rng(1337)
    eta = (rng.normal(size=(5, 2)) @ rng.normal(size=(2, 4))).astype(np.complex128)

    u, s, vh, keep = _thin_svd_and_rank(eta, 1e-10)
    kept = np.flatnonzero(keep)

    assert keep.tolist() == [True, True, False, False]
    assert_allclose(u[:, kept] @ np.diag(s[kept]) @ vh[kept], eta, atol=1e-12)


def test_split_matrix_on_eigen_stability():
    Q = np.arange(25, dtype=float).reshape(5, 5)
    n_unstable = 3

    Q1, Q2 = split_matrix_on_eigen_stability(Q, n_unstable)

    assert_allclose(Q1, Q[:2, :])
    assert_allclose(Q2, Q[2:, :])


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
