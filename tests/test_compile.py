from typing import Literal

import numpy as np
import pytest
import sympy as sp

from gEconpy.model.compile import BACKENDS, compile_function


@pytest.mark.parametrize("backend", ["numpy", "numba", "pytensor"])
def test_scalar_function(backend: Literal["numpy", "numba", "pytensor"]):
    x = sp.symbols("x")
    f = x**2
    f_func, _ = compile_function(
        [x], f, backend=backend, mode="FAST_COMPILE", pop_return=backend == "pytensor"
    )
    assert f_func(x=2) == 4


@pytest.mark.parametrize("backend", ["numpy", "numba", "pytensor"])
@pytest.mark.parametrize("stack_return", [True, False])
def test_multiple_outputs(
    backend: Literal["numpy", "numba", "pytensor"], stack_return: bool
):
    x, y, z = sp.symbols("x y z ")
    x2 = x**2
    y2 = y**2
    z2 = z**2
    f_func, _ = compile_function(
        [x, y, z],
        [x2, y2, z2],
        backend=backend,
        stack_return=stack_return,
        mode="FAST_COMPILE",
    )
    res = f_func(x=2, y=3, z=4)
    assert (
        isinstance(res, np.ndarray) if stack_return else isinstance(res, list | tuple)
    )
    assert res.shape == (3,) if stack_return else len(res) == 3
    np.testing.assert_allclose(
        res if stack_return else np.stack(res), np.array([4.0, 9.0, 16.0])
    )


@pytest.mark.parametrize("backend", ["numpy", "numba", "pytensor"])
def test_matrix_function(backend: Literal["numpy", "numba", "pytensor"]):
    x, y, z = sp.symbols("x y z")
    f = sp.Matrix([x, y, z]).reshape(1, 3)

    f_func, _ = compile_function(
        [x, y, z],
        f,
        backend=backend,
        mode="FAST_COMPILE",
        pop_return=backend == "pytensor",
    )
    res = f_func(x=2, y=3, z=4)

    assert isinstance(res, np.ndarray)
    assert res.shape == (1, 3)
    np.testing.assert_allclose(res, np.array([[2.0, 3.0, 4.0]]))


@pytest.mark.parametrize("backend", ["numpy", "numba", "pytensor"])
def test_compile_gradient(backend: BACKENDS):
    x, y, z = sp.symbols("x y z")
    f = x**2 + y**2 + z**2
    grad = sp.Matrix([f.diff(x), f.diff(y), f.diff(z)]).reshape(3, 1)
    grad_func, _ = compile_function(
        [x, y, z],
        grad,
        backend=backend,
        mode="FAST_COMPILE",
        pop_return=backend == "pytensor",
    )
    res = grad_func(x=2.0, y=3.0, z=4.0)
    np.testing.assert_allclose(res, np.array([4.0, 6.0, 8.0])[:, None])

    hess = grad.jacobian([x, y, z])
    hess_func, _ = compile_function(
        [x, y, z],
        hess,
        backend=backend,
        mode="FAST_COMPILE",
        pop_return=backend == "pytensor",
    )
    res = hess_func(x=2.0, y=3.0, z=4.0)
    np.testing.assert_allclose(res, np.eye(3) * 2.0)
