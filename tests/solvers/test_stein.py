import importlib.util

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose

from gEconpy.solvers.cycle_reduction import cycle_reduction_numpy
from gEconpy.solvers.stein import ADJOINT_METHODS, solve_policy_stein
from tests._resources.cache_compiled_models import load_and_cache_model

requires_jax = pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="jax not installed")
BACKENDS = ["NUMBA", "CVM", pytest.param("JAX", marks=requires_jax)]


def stein_problem(n=6, seed=0):
    """Build a random Stein equation A1 S + C' S T' = Q with rho(A1^{-1} C') rho(T) < 1, so every method converges."""
    rng = np.random.default_rng(seed)
    A1 = rng.normal(size=(n, n)) + n * np.eye(n)
    C = 0.5 * rng.normal(size=(n, n))
    T = rng.normal(size=(n, n))
    T *= 0.8 / np.abs(np.linalg.eigvals(T)).max()
    Q = rng.normal(size=(n, n))
    return A1, C, T, Q


def stein_symbols(dtype="float64"):
    return [pt.matrix(name, dtype=dtype) for name in ("A1", "C", "T", "Q")]


# PyTensor has no JAX implementation of QZ.
METHOD_BACKENDS = [
    pytest.param(method, mode, marks=requires_jax if mode == "JAX" else (), id=f"{method}-{mode}")
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
