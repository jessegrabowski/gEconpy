import numpy as np
import pytensor.tensor as pt
import pytest
import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.model.compile import (
    compile_function,
    dictionary_return_wrapper,
    make_return_dict_and_update_cache,
    sympy_to_pytensor,
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


def test_scalar_function():
    x = sp.symbols("x")
    f = x**2
    f_func, _ = compile_function([x], f, mode="FAST_COMPILE", pop_return=True)
    result = f_func(x=np.float64(2))
    np.testing.assert_allclose(result, 4.0)


@pytest.mark.parametrize("stack_return", [True, False])
def test_multiple_outputs(stack_return: bool):
    x, y, z = sp.symbols("x y z")
    x2 = x**2
    y2 = y**2
    z2 = z**2
    f_func, _ = compile_function(
        [x, y, z],
        [x2, y2, z2],
        stack_return=stack_return,
        mode="FAST_COMPILE",
    )
    res = f_func(x=np.float64(2), y=np.float64(3), z=np.float64(4))
    if stack_return:
        assert isinstance(res, np.ndarray)
        assert res.shape == (3,)
        np.testing.assert_allclose(res, np.array([4.0, 9.0, 16.0]))
    else:
        assert isinstance(res, list)
        assert len(res) == 3
        np.testing.assert_allclose(np.stack(res), np.array([4.0, 9.0, 16.0]))


def test_matrix_function():
    x, y, z = sp.symbols("x y z")
    f = sp.Matrix([x, y, z]).reshape(1, 3)

    f_func, _ = compile_function(
        [x, y, z],
        f,
        mode="FAST_COMPILE",
        pop_return=True,
    )
    res = f_func(x=np.float64(2), y=np.float64(3), z=np.float64(4))

    assert isinstance(res, np.ndarray)
    assert res.shape == (1, 3)
    np.testing.assert_allclose(res, np.array([[2.0, 3.0, 4.0]]))


def test_compile_gradient():
    x, y, z = sp.symbols("x y z")
    f = x**2 + y**2 + z**2
    grad = sp.Matrix([f.diff(x), f.diff(y), f.diff(z)]).reshape(3, 1)
    grad_func, _ = compile_function(
        [x, y, z],
        grad,
        mode="FAST_COMPILE",
        pop_return=True,
    )
    res = grad_func(x=np.float64(2.0), y=np.float64(3.0), z=np.float64(4.0))
    np.testing.assert_allclose(res, np.array([4.0, 6.0, 8.0])[:, None])

    hess = grad.jacobian([x, y, z])
    hess_func, _ = compile_function(
        [x, y, z],
        hess,
        mode="FAST_COMPILE",
        pop_return=True,
    )
    res = hess_func(x=np.float64(2.0), y=np.float64(3.0), z=np.float64(4.0))
    np.testing.assert_allclose(res, np.eye(3) * 2.0)


def test_sympy_to_pytensor():
    x, y = sp.symbols("x y")
    z_expr = x**2 + y

    input_nodes, output_nodes, cache = sympy_to_pytensor([x, y], [z_expr])

    assert len(input_nodes) == 2
    assert len(output_nodes) == 1
    assert all(hasattr(n, "type") for n in input_nodes)
    assert all(hasattr(n, "type") for n in output_nodes)
    assert len(cache) > 0


def test_sympy_to_pytensor_shared_cache():
    x = sp.symbols("x")
    cache = {}

    _, _, cache = sympy_to_pytensor([x], [x**2], cache)
    _, out2, cache = sympy_to_pytensor([x], [x**3], cache)

    # Second call should reuse the same input node from cache
    assert len(out2) == 1


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
