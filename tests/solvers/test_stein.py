import importlib.util

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pytensor.gradient import verify_grad
from threadpoolctl import threadpool_info

from gEconpy.solvers.cycle_reduction import cycle_reduction_numpy
from gEconpy.solvers.stein import ADJOINT_METHODS, solve_generalized_sylvester, solve_policy_stein
from tests._resources.cache_compiled_models import load_and_cache_model

requires_jax = pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="jax not installed")
# JAX's CPU Schur decomposition, which jax.scipy.linalg.solve_sylvester calls, segfaults when LAPACK is MKL.
skip_jax_schur_under_mkl = pytest.mark.skipif(
    any(info["internal_api"] == "mkl" for info in threadpool_info()),
    reason="JAX's CPU Schur decomposition segfaults with MKL",
)
BACKENDS = ["NUMBA", "CVM", pytest.param("JAX", marks=requires_jax)]


def stein_problem(n=6, n_columns=None, seed=0):
    """Build a random Stein equation A1 S + C' S T' = Q with rho(A1^{-1} C') rho(T) < 1, so every method converges."""
    n_columns = n if n_columns is None else n_columns
    rng = np.random.default_rng(seed)
    A1 = rng.normal(size=(n, n)) + n * np.eye(n)
    C = 0.5 * rng.normal(size=(n, n))
    T = rng.normal(size=(n_columns, n_columns))
    T *= 0.8 / np.abs(np.linalg.eigvals(T)).max()
    Q = rng.normal(size=(n, n_columns))
    return A1, C, T, Q


def stein_symbols(dtype="float64"):
    return [pt.matrix(name, dtype=dtype) for name in ("A1", "C", "T", "Q")]


JAX_MARKS = {"sylvester": [requires_jax, skip_jax_schur_under_mkl]}

# PyTensor has no JAX implementation of QZ.
METHOD_BACKENDS = [
    pytest.param(
        method, mode, marks=JAX_MARKS.get(method, requires_jax) if mode == "JAX" else (), id=f"{method}-{mode}"
    )
    for method in ADJOINT_METHODS
    for mode in ("NUMBA", "CVM", "JAX")
    if (method, mode) != ("qz", "JAX")
]


@pytest.mark.parametrize(("dtype", "atol"), [("float64", 1e-10), ("float32", 1e-4)], ids=["float64", "float32"])
@pytest.mark.parametrize(("method", "mode"), METHOD_BACKENDS)
def test_solution_satisfies_stein_equation(method, mode, dtype, atol):
    A1, C, T, Q = (x.astype(dtype) for x in stein_problem())
    symbols = stein_symbols(dtype)
    S = pytensor.function(symbols, solve_policy_stein(*symbols, method=method), mode=mode)(A1, C, T, Q)

    assert S.dtype == dtype
    assert_allclose(A1 @ S + C.T @ S @ T.T, Q, atol=atol)


@pytest.mark.parametrize("n_columns", [3, 9], ids=["narrow", "wide"])
@pytest.mark.parametrize(("method", "mode"), METHOD_BACKENDS)
def test_rectangular_solution_satisfies_stein_equation(method, mode, n_columns):
    A1, C, T, Q = stein_problem(n_columns=n_columns)
    symbols = stein_symbols()
    S = pytensor.function(symbols, solve_policy_stein(*symbols, method=method), mode=mode)(A1, C, T, Q)

    assert S.shape == Q.shape
    assert_allclose(A1 @ S + C.T @ S @ T.T, Q, atol=1e-10)


def rotation(angle, radius):
    return radius * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def sylvester_problem(K, n=5, seed=0):
    """Build A X + B X K = D from a known X, with rho(A^{-1} B) rho(K) < 1 so every method converges."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + n * np.eye(n)
    B = 0.5 * rng.normal(size=(n, n))
    X = rng.normal(size=(n, K.shape[0]))
    return A, B, K, A @ X + B @ X @ K, X


SYLVESTER_COEFFICIENTS = {
    "power_one": np.diag([0.5, -0.3, 0.7]) + np.triu(np.full((3, 3), 0.2), k=1),
    "kronecker_square": np.kron(rotation(0.4, 0.9), np.array([[0.6, 0.3], [0.0, -0.5]])),
    "kronecker_cube": np.kron(np.kron(rotation(1.1, 0.8), rotation(0.3, 0.7)), np.diag([0.9, 0.2])),
}


@pytest.mark.parametrize("K", SYLVESTER_COEFFICIENTS.values(), ids=SYLVESTER_COEFFICIENTS.keys())
@pytest.mark.parametrize("method", ADJOINT_METHODS)
def test_generalized_sylvester_recovers_a_known_solution(method, K):
    A, B, K, D, X = sylvester_problem(K)
    symbols = [pt.dmatrix(name) for name in "ABKD"]
    solution = pytensor.function(symbols, solve_generalized_sylvester(*symbols, method=method))(A, B, K, D)

    assert_allclose(solution, X, atol=1e-10)


@pytest.mark.parametrize("method", ["doubling", "sylvester"])
def test_generalized_sylvester_gradient_matches_finite_differences(method):
    A, B, K, D, _ = sylvester_problem(SYLVESTER_COEFFICIENTS["kronecker_square"])
    rng = np.random.default_rng(1)

    verify_grad(lambda A, B, K, D: solve_generalized_sylvester(A, B, K, D, method=method), pt=[A, B, K, D], rng=rng)


@pytest.mark.include_nk
def test_methods_agree_on_a_dsge_model():
    mod = load_and_cache_model("full_nk.gcn")
    A, B, C, _ = mod.linearize_model(verbose=False, steady_state_kwargs={"verbose": False, "progressbar": False})
    T, *_ = cycle_reduction_numpy(A, B, C, max_iter=1000, tol=1e-12)
    Q = np.random.default_rng(0).normal(size=T.shape)

    symbols = stein_symbols()
    solutions = {
        method: pytensor.function(symbols, solve_policy_stein(*symbols, method=method))((B + C @ T).T, C, T, Q)
        for method in ADJOINT_METHODS
    }

    reference = solutions["direct"]
    for method in ("doubling", "qz", "sylvester"):
        assert_allclose(solutions[method], reference, atol=1e-10 * np.abs(reference).max(), err_msg=method)


@pytest.mark.parametrize("mode", BACKENDS)
def test_doubling_returns_nan_when_the_series_does_not_converge(mode):
    """With rho(M) rho(T) within 1e-9 of one, 30 doubling steps cover only part of the series."""
    M = np.diag([1 - 1e-9, 0.1, 0.2, 0.3])
    A1, C, T, Q = np.eye(4), -M.T, np.eye(4), np.ones((4, 4))
    symbols = stein_symbols()
    S = pytensor.function(symbols, solve_policy_stein(*symbols, method="doubling"), mode=mode)(A1, C, T, Q)

    assert np.isnan(S).all()


@pytest.mark.parametrize("mode", ["NUMBA", "CVM"])
def test_qz_returns_nan_when_the_equation_is_singular(mode):
    """An eigenvalue of M times one of T equal to -1 leaves the Stein equation without a unique solution."""
    A1, C, T, Q = np.eye(2), np.eye(2), -np.eye(2), np.ones((2, 2))
    symbols = stein_symbols()
    S = pytensor.function(symbols, solve_policy_stein(*symbols, method="qz"), mode=mode)(A1, C, T, Q)

    assert np.isnan(S).all()
