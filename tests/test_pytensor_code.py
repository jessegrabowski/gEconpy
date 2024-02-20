import numpy as np
import pytensor
import sympy as sp

from numpy.testing import assert_allclose

from gEconpy.sympy_tools.pytensorcode import pytensor_code


def test_write_to_pytensor_basic():
    x, y, z = sp.symbols("x y z")
    z = x**2 + y + x * y * z
    cache = {}
    z_pt = pytensor_code(z, cache=cache)
    assert len(cache) == 3

    inputs = list(cache.values())
    f = pytensor.function(inputs, [z_pt])

    assert f(1, 1, 1)[0] == 3


def test_dot_from_symbolic():
    A = sp.MatrixSymbol("A", 5, 5)
    b = sp.MatrixSymbol("b", 5, 1)

    B = A @ b
    cache = {}
    B_pt = pytensor_code(B, cache=cache)
    inputs = list(cache.values())
    f = pytensor.function(inputs, [B_pt])

    assert_allclose(f(np.eye(5), np.ones((5, 1)))[0], np.ones((5, 1)))


def test_det_from_symbolic():
    A = sp.MatrixSymbol("A", 5, 5)
    d = sp.Determinant(A)
    cache = {}
    d_pt = pytensor_code(d, cache=cache)
    inputs = list(cache.values())

    f = pytensor.function(inputs, [d_pt])

    x = np.random.normal(size=(3, 3)) ** 2
    assert_allclose(f(x)[0], np.linalg.det(x))


def test_inv_from_symbolic():
    A = sp.MatrixSymbol("A", 5, 5)
    A_inv = sp.Inverse(A)
    cache = {}
    A_inv_pt = pytensor_code(A_inv, cache=cache)
    inputs = list(cache.values())
    f = pytensor.function(inputs, [A_inv_pt])

    x = np.random.normal(size=(3, 3))
    x = x @ x.T
    assert_allclose(f(x)[0], np.linalg.inv(x))


def test_convert_constant_value():
    A = sp.Float(1.0)
    cache = {}
    A_pt = pytensor_code(A, cache=cache)
    inputs = list(cache.values())
    f = pytensor.function(inputs, [A_pt])
    assert f()[0] == 1.0
