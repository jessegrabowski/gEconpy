import numpy as np
import pytensor
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
    """Start each test with a fresh compile cache."""
    clear_compile_cache()
    yield
    clear_compile_cache()


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


def test_givens_as_dict_and_list_share_a_cache_entry():
    x = pt.dscalar("x")
    y = pt.dscalar("y")
    z = x + y

    two = pt.constant(2.0, dtype="float64")
    f_dict = compile_pytensor_function([x], [z], givens={y: two}, mode="FAST_COMPILE")
    f_list = compile_pytensor_function([x], [z], givens=[(y, two)], mode="FAST_COMPILE")

    assert f_dict is f_list
    np.testing.assert_allclose(f_dict(np.float64(1.0)), [3.0])


def test_clear_cache():
    x = pt.dscalar("x")
    z = x**2

    compile_pytensor_function([x], [z], mode="FAST_COMPILE")
    compile_pytensor_function([x], [z], mode="FAST_COMPILE")
    before = compile_cache_info()
    assert (before.hits, before.misses, before.currsize) == (1, 1, 1)

    clear_compile_cache()
    after = compile_cache_info()
    assert (after.hits, after.misses, after.currsize) == (0, 0, 0)


def test_a_rebuilt_graph_hits_the_cache():
    """
    A graph rebuilt from scratch hits the cache, which is the whole point of keying on its structure.

    Nothing in a model holds its graph nodes between calls, so a cache keyed on node identity never hits and
    every call recompiles.
    """

    def build():
        x = pt.dscalar("x")
        y = pt.dscalar("y")
        return [x, y], [x * y + pt.exp(x)]

    f1 = compile_pytensor_function(*build(), mode="FAST_COMPILE")
    f2 = compile_pytensor_function(*build(), mode="FAST_COMPILE")

    assert f1 is f2
    assert compile_cache_info().hits == 1
    np.testing.assert_allclose(f2(np.float64(1.0), np.float64(2.0)), [2.0 + np.exp(1.0)])


def test_inputs_named_differently_do_not_share_an_entry():
    """Callers read these functions by keyword, so a function whose inputs carry other names cannot stand in."""
    a, b = pt.dscalar("alpha"), pt.dscalar("beta")
    p, q = pt.dscalar("phi"), pt.dscalar("psi")

    f_greek = compile_pytensor_function([a, b], [a * b], mode="FAST_COMPILE")
    f_other = compile_pytensor_function([p, q], [p * q], mode="FAST_COMPILE")

    assert f_greek is not f_other
    assert {inp.name for inp in f_other.input_storage} == {"phi", "psi"}


def test_inputs_in_a_different_order_do_not_share_an_entry():
    """The two functions take their arguments positionally, so swapping the inputs is a different function."""
    x, y = pt.dscalar("x"), pt.dscalar("y")

    forward = compile_pytensor_function([x, y], [x - y], mode="FAST_COMPILE")
    reversed_ = compile_pytensor_function([y, x], [x - y], mode="FAST_COMPILE")

    assert forward is not reversed_
    np.testing.assert_allclose(forward(np.float64(5.0), np.float64(3.0)), [2.0])
    np.testing.assert_allclose(reversed_(np.float64(5.0), np.float64(3.0)), [-2.0])


def test_a_differing_constant_does_not_share_an_entry():
    """
    A constant is baked into the compiled function, so two graphs differing only there are different.

    The array is deliberately long. A key built by printing the graph abbreviates a large constant, which
    makes the two graphs look identical, so a short array here would stop guarding anything.
    """

    def build(weights):
        x = pt.dvector("x")
        return [x], [(x * pt.as_tensor_variable(weights)).sum()]

    first = np.arange(500, dtype=float)
    second = first.copy()
    second[400] = -99.0

    f1 = compile_pytensor_function(*build(first), mode="FAST_COMPILE")
    f2 = compile_pytensor_function(*build(second), mode="FAST_COMPILE")

    assert f1 is not f2
    np.testing.assert_allclose(f1(np.ones(500)), [first.sum()])
    np.testing.assert_allclose(f2(np.ones(500)), [second.sum()])


def test_a_graph_holding_a_shared_variable_is_cached():
    """A shared variable is not listed as an input, so a key built from the inputs alone cannot be made at all."""
    counter = pytensor.shared(np.float64(1.0), name="counter")
    x = pt.dscalar("x")

    f1 = compile_pytensor_function([x], [x + counter], mode="FAST_COMPILE")
    f2 = compile_pytensor_function([x], [x + counter], mode="FAST_COMPILE")

    assert f1 is f2
    assert compile_cache_info().hits == 1


def test_different_shared_variables_do_not_share_an_entry():
    """Freezing compares a shared variable by type, so two holding different values look identical."""
    first = pytensor.shared(np.float64(1.0), name="s")
    second = pytensor.shared(np.float64(2.0), name="s")
    x = pt.dscalar("x")

    f_first = compile_pytensor_function([x], [x + first], mode="FAST_COMPILE")
    f_second = compile_pytensor_function([x], [x + second], mode="FAST_COMPILE")

    assert f_first is not f_second
    np.testing.assert_allclose(f_first(np.float64(10.0)), [11.0])
    np.testing.assert_allclose(f_second(np.float64(10.0)), [12.0])


def test_differing_updates_do_not_share_an_entry():
    """An update writes to a shared variable, so two functions writing different values are not the same."""
    counter = pytensor.shared(np.float64(0.0), name="counter")
    x = pt.dscalar("x")

    increment = compile_pytensor_function([x], [x + counter], updates=[(counter, counter + 1)], mode="FAST_COMPILE")
    decrement = compile_pytensor_function([x], [x + counter], updates=[(counter, counter - 1)], mode="FAST_COMPILE")

    assert increment is not decrement

    increment(np.float64(1.0))
    assert counter.get_value() == 1.0
    decrement(np.float64(1.0))
    assert counter.get_value() == 0.0


def test_one_output_and_a_sequence_of_one_do_not_share_an_entry():
    """``pytensor.function`` returns a bare array for one output and a list for a sequence of them."""
    x = pt.dscalar("x")

    bare = compile_pytensor_function([x], x**2, mode="FAST_COMPILE")
    sequence = compile_pytensor_function([x], [x**2], mode="FAST_COMPILE")

    assert bare is not sequence
    np.testing.assert_allclose(bare(np.float64(3.0)), 9.0)
    np.testing.assert_allclose(sequence(np.float64(3.0)), [9.0])
