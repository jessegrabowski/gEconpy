import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor.graph.replace import graph_replace

from gEconpy.pytensorf.compile import (
    clear_compile_cache,
    compile_cache_info,
    compile_pytensor_function,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure each test starts with a fresh compile cache."""
    clear_compile_cache()
    yield
    clear_compile_cache()


def test_cache_hits():
    x = pt.dscalar("x")
    z = x**2

    f1 = compile_pytensor_function([x], [z], mode="FAST_COMPILE")
    f2 = compile_pytensor_function([x], [z], mode="FAST_COMPILE")

    assert f1 is f2
    assert compile_cache_info().hits >= 1


def test_cache_miss_different_mode():
    x = pt.dscalar("x")
    z = x**2

    f1 = compile_pytensor_function([x], [z], mode="FAST_COMPILE")
    f2 = compile_pytensor_function([x], [z], mode="FAST_RUN")

    assert f1 is not f2


def test_cache_miss_after_graph_replace():
    x = pt.dscalar("x")
    y = pt.dscalar("y")
    z = x + y

    f1 = compile_pytensor_function([x, y], [z], mode="FAST_COMPILE")

    z2 = graph_replace(z, {x: pt.exp(x)})
    f2 = compile_pytensor_function([x, y], [z2], mode="FAST_COMPILE")

    assert f1 is not f2

    np.testing.assert_allclose(f1(np.float64(1.0), np.float64(2.0)), [3.0])
    np.testing.assert_allclose(f2(np.float64(1.0), np.float64(2.0)), [np.exp(1.0) + 2.0])


def test_clear_cache():
    x = pt.dscalar("x")
    z = x**2

    compile_pytensor_function([x], [z], mode="FAST_COMPILE")
    assert compile_cache_info().currsize == 1

    clear_compile_cache()
    assert compile_cache_info().currsize == 0
