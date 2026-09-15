import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import (
    build_symbolic_jacobians,
    compile_for_scipy,
    compile_function,
    dictionary_return_wrapper,
    make_cache_key,
    make_return_dict_and_update_cache,
    pack_and_compile,
    sympy_to_pytensor,
)


def test_dictionary_return_wrapper():
    outputs = [sp.Symbol("a"), sp.Symbol("b")]

    def f():
        return [1.0, 2.0]

    wrapped = dictionary_return_wrapper(f, outputs)
    result = wrapped()

    assert isinstance(result, SymbolDictionary)
    assert result["a"] == 1.0
    assert result["b"] == 2.0


def test_scalar_function():
    x = sp.symbols("x")
    f = x**2
    f_func, _ = compile_function([x], f, mode="FAST_COMPILE", pop_return=True)
    result = f_func(x=np.float64(2))
    np.testing.assert_allclose(result, 4.0)


@pytest.mark.parametrize("stack_return, expected_type", [(True, np.ndarray), (False, list)], ids=["stacked", "list"])
def test_multiple_outputs(stack_return: bool, expected_type: type):
    x, y, z = sp.symbols("x y z")
    f_func, _ = compile_function(
        [x, y, z],
        [x**2, y**2, z**2],
        stack_return=stack_return,
        mode="FAST_COMPILE",
    )
    res = f_func(x=np.float64(2), y=np.float64(3), z=np.float64(4))

    assert isinstance(res, expected_type)
    np.testing.assert_allclose(np.asarray(res), np.array([4.0, 9.0, 16.0]))


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


@pytest.mark.parametrize("cse", [False, True], ids=["no_cse", "cse"])
def test_sympy_to_pytensor_evaluates_outputs_and_caches_inputs(cse):
    x, y = sp.symbols("x y")
    shared = (x + y) ** 2

    input_nodes, output_nodes, cache = sympy_to_pytensor([x, y], [shared * x, shared * y, 3.0], cse=cse)

    assert [cache[make_cache_key(name, sp.Symbol)] for name in ("x", "y")] == input_nodes
    f = pytensor.function(input_nodes, output_nodes, mode="FAST_COMPILE")
    np.testing.assert_allclose(f(2.0, 1.0), [18.0, 9.0, 3.0])


def test_sympy_to_pytensor_shared_cache():
    x = sp.symbols("x")
    cache = {}

    (x_pt_first,), _, cache = sympy_to_pytensor([x], [x**2], cache)
    (x_pt_second,), _, cache = sympy_to_pytensor([x], [x**3], cache)

    assert x_pt_second is x_pt_first


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


def test_compile_for_scipy_ignores_unknown_keywords():
    x, y = pt.dscalars("x", "y")
    f = compile_for_scipy(x * y, mode="FAST_COMPILE")

    np.testing.assert_allclose(f(x=2.0, y=3.0, z=100.0), 6.0)


@pytest.mark.parametrize("freeze_params", [True, False], ids=["frozen_params", "positional_params"])
def test_pack_and_compile_reads_only_active_nodes_from_flat_vector(freeze_params):
    x, y, z, alpha = pt.dscalars("x", "y", "z", "alpha")
    param_dict = SymbolDictionary({"alpha": 10.0}) if freeze_params else None
    param_args = () if freeze_params else (10.0,)

    # y is not a graph input, so x_flat[1] is skipped and z is read from x_flat[2].
    f = pack_and_compile(pt.stack([x * alpha, z + alpha]), [x, y, z], param_dict=param_dict, mode="FAST_COMPILE")

    np.testing.assert_allclose(f(np.array([1.0, 999.0, 2.0]), *param_args), [10.0, 12.0])


@pytest.mark.parametrize("freeze_params", [True, False], ids=["frozen_params", "positional_params"])
def test_pack_and_compile_without_steady_state_inputs_ignores_flat_vector(freeze_params):
    x, alpha = pt.dscalars("x", "alpha")
    param_dict = SymbolDictionary({"alpha": 10.0, "unused": 1.0}) if freeze_params else None
    param_args = () if freeze_params else (10.0,)

    f = pack_and_compile(alpha * 2.0, [x], param_dict=param_dict, mode="FAST_COMPILE")

    np.testing.assert_allclose(f(np.array([123.0]), *param_args), 20.0)


def test_build_symbolic_jacobians_at_steady_state_shares_one_cache():
    x, y = TimeAwareSymbol("x", 0), TimeAwareSymbol("y", 0)
    eps = TimeAwareSymbol("eps", 0)
    alpha = sp.Symbol("alpha")
    equations = [x - alpha * y.set_t(-1) ** 2 - eps, y - sp.log(x.set_t(1))]

    cache = {}
    jac_lag, jac_lead, jac_empty = build_symbolic_jacobians(
        [(equations, [x.set_t(-1), y.set_t(-1)]), (equations, [x.set_t(1), y.set_t(1)]), (equations, [])],
        cache,
        to_ss=True,
        shocks=[eps],
    )

    inputs = [
        cache[make_cache_key(name, cls)]
        for name, cls in [("x_ss", TimeAwareSymbol), ("y_ss", TimeAwareSymbol), ("alpha", sp.Symbol)]
    ]
    f = pytensor.function(inputs, [jac_lag, jac_lead], mode="FAST_COMPILE", on_unused_input="ignore")
    lag_value, lead_value = f(2.0, 3.0, 0.5)

    np.testing.assert_allclose(lag_value, [[0.0, -2 * 0.5 * 3.0], [0.0, 0.0]])
    np.testing.assert_allclose(lead_value, [[0.0, 0.0], [-1 / 2.0, 0.0]])
    assert jac_empty.type.shape == (2, 0)
