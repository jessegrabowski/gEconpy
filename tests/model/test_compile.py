from typing import Literal

import numpy as np
import pytensor.tensor as pt
import pytest
import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.model.compile import (
    BACKENDS,
    array_return_wrapper,
    compile_function,
    dictionary_return_wrapper,
    make_return_dict_and_update_cache,
    pop_return_wrapper,
    stack_return_wrapper,
)


def test_dictionary_return_wrapper():
    outputs = [sp.Symbol("a"), sp.Symbol("b")]

    def f():
        return [1.0, 2.0]

    wrapped = dictionary_return_wrapper(f, outputs)
    result = wrapped()

    assert isinstance(result, SymbolDictionary)
    assert "a" in result and "b" in result
    assert result["a"] == 1.0
    assert result["b"] == 2.0


@pytest.mark.parametrize("backend", ["numpy", "pytensor"])
def test_scalar_function(backend: Literal["numpy", "pytensor"]):
    x = sp.symbols("x")
    f = x**2
    f_func, _ = compile_function([x], f, backend=backend, mode="FAST_COMPILE", pop_return=backend == "pytensor")
    assert f_func(x=2) == 4


@pytest.mark.parametrize("backend", ["numpy", "pytensor"])
@pytest.mark.parametrize("stack_return", [True, False])
def test_multiple_outputs(backend: Literal["numpy", "pytensor"], stack_return: bool):
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
    assert isinstance(res, np.ndarray) if stack_return else isinstance(res, list | tuple)
    assert res.shape == (3,) if stack_return else len(res) == 3
    np.testing.assert_allclose(res if stack_return else np.stack(res), np.array([4.0, 9.0, 16.0]))


@pytest.mark.parametrize("backend", ["numpy", "pytensor"])
def test_matrix_function(backend: Literal["numpy", "pytensor"]):
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


@pytest.mark.parametrize("backend", ["numpy", "pytensor"])
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


def test_compile_function_invalid_backend():
    x = sp.symbols("x")
    with pytest.raises(NotImplementedError):
        compile_function([x], x**2, backend="invalid_backend")


def test_stack_return_wrapper_list():
    def f(x):
        return [x, x + 1, x + 2]

    wrapped = stack_return_wrapper(f)
    res = wrapped(1)
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, np.array([1, 2, 3]))


def test_stack_return_wrapper_scalar():
    def f(x):
        return x + 5

    wrapped = stack_return_wrapper(f)
    res = wrapped(2)
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, np.array([7]))


def test_pop_return_wrapper_scalar():
    def f(x):
        return [x + 10]

    wrapped = pop_return_wrapper(f)
    res = wrapped(3)
    assert res == 13


def test_pop_return_wrapper_array():
    def f(x):
        return [x, x + 1]

    wrapped = pop_return_wrapper(f)
    res = wrapped(4)
    assert res == 4


def test_array_return_wrapper():
    def f(x):
        return [x, x + 1]

    wrapped = array_return_wrapper(f)
    res = wrapped(5)
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, np.array([5, 6]))


def test_make_return_dict_and_update_cache():
    x, y = sp.symbols("x y")
    x_pt, y_pt = pt.dscalars("x", "y")

    cache = {}
    out_dict, new_cache = make_return_dict_and_update_cache([x, y], [x_pt, y_pt], cache)

    assert x_pt in out_dict.values()
    assert y_pt in out_dict.values()
    assert len(new_cache) == 2
    assert all(isinstance(k, tuple) for k in new_cache)
    assert all(hasattr(v, "type") for v in new_cache.values())

    z = sp.symbols("z")
    z_pt = pt.dscalar("z")

    out_dict, new_cache_2 = make_return_dict_and_update_cache([x, y, z], [x_pt, y_pt, z_pt], cache.copy())
    x_key, *_ = new_cache.keys()
    assert z_pt in out_dict.values()
    assert new_cache_2[x_key] is new_cache[x_key]
