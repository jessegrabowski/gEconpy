from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
import sympy as sp
from pytensor.tensor import TensorVariable

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.parser.file_loaders import gcn_to_block_dict, block_dict_to_param_dict


@pytest.fixture
def simple_param_system():
    alpha, beta, gamma = sp.symbols("alpha beta gamma")
    param_dict = SymbolDictionary({alpha: 0.3, beta: 0.99})
    deterministic_dict = SymbolDictionary({gamma: alpha + beta})
    return param_dict, deterministic_dict


@pytest.fixture
def complex_param_system():
    alpha, beta, gamma, delta, theta = sp.symbols("alpha beta gamma delta theta")
    param_dict = SymbolDictionary({alpha: 0.3, beta: 0.99, theta: 0.5})

    gamma_val = sp.log(alpha + beta)
    deterministic_dict = SymbolDictionary({gamma: gamma_val, delta: gamma_val * theta})

    return param_dict, deterministic_dict


@pytest.mark.parametrize("backend", ["numpy", "pytensor"])
def test_compile_param_dict_basic(simple_param_system, backend):
    param_dict, deterministic_dict = simple_param_system

    f, _ = compile_param_dict_func(param_dict, deterministic_dict, backend=backend)
    result = f(alpha=0.3, beta=0.99)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"alpha", "beta", "gamma"}
    assert_allclose(result["gamma"], 0.3 + 0.99)


@pytest.mark.parametrize("backend", ["numpy", "pytensor"])
def test_compile_param_dict_complex(complex_param_system, backend):
    param_dict, deterministic_dict = complex_param_system

    f, _ = compile_param_dict_func(param_dict, deterministic_dict, backend=backend)
    result = f(alpha=0.3, beta=0.99, theta=0.5)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"alpha", "beta", "gamma", "delta", "theta"}
    expected_gamma = np.log(0.3 + 0.99)
    assert_allclose(result["gamma"], expected_gamma)
    assert_allclose(result["delta"], expected_gamma * 0.5)


@pytest.mark.parametrize("backend", ["numpy", "pytensor"])
def test_compile_param_dict_cache_reuse(complex_param_system, backend):
    param_dict, deterministic_dict = complex_param_system

    # First compilation should create cache
    cache = {}
    f1, cache = compile_param_dict_func(
        param_dict, deterministic_dict, backend=backend, cache=cache
    )

    # Second compilation should reuse cache
    f2, cache2 = compile_param_dict_func(
        param_dict, deterministic_dict, backend=backend, cache=cache
    )

    # Results should be identical
    result1 = f1(alpha=0.3, beta=0.99, theta=0.5)
    result2 = f2(alpha=0.3, beta=0.99, theta=0.5)

    assert cache is cache2  # Same cache object
    assert_allclose(np.array(list(result1.values())), np.array(list(result2.values())))


def test_compile_param_dict_symbolic(complex_param_system):
    param_dict, deterministic_dict = complex_param_system

    symbolic_result, cache = compile_param_dict_func(
        param_dict, deterministic_dict, backend="pytensor", return_symbolic=True
    )

    assert isinstance(symbolic_result, dict)
    assert isinstance(cache, dict)
    assert all(isinstance(k, TensorVariable) for k in symbolic_result.keys())

    *_, gamma, delta = symbolic_result.keys()
    np.testing.assert_allclose(
        symbolic_result[gamma].eval({"alpha": 2.0, "beta": 3.0}), np.log(5.0)
    )


EXPECTED_PARAM_DICT = {
    "one_block_simple": dict(alpha=0.4, beta=0.99, delta=0.02, rho=0.95, gamma=1.5),
    "one_block_simple_2": dict(
        theta=0.357,
        beta=1 / 1.01,
        delta=0.02,
        tau=2,
        rho=0.95,
        Theta=0.95 * 1 / 1.01 + 3,
        zeta=-np.log(0.357),
    ),
}


@pytest.mark.parametrize(
    "gcn_path, name",
    [
        ("one_block_1.gcn", "one_block_simple"),
        ("one_block_2.gcn", "one_block_simple_2"),
    ],
    ids=["one_block_simple", "one_block_simple_2"],
)
@pytest.mark.parametrize("backend", ["numpy", "pytensor"], ids=["numpy", "pytensor"])
def test_create_parameter_function(gcn_path, name, backend):
    expected = EXPECTED_PARAM_DICT[name]
    block_dict, *outputs = gcn_to_block_dict(
        Path("tests") / "_resources" / "test_gcns" / gcn_path, True
    )
    param_dict = block_dict_to_param_dict(block_dict, "param_dict")
    deterministic_dict = block_dict_to_param_dict(block_dict, "deterministic_dict")

    f, _ = compile_param_dict_func(param_dict, deterministic_dict, backend)

    inputs = list(param_dict.keys())
    np.random.shuffle(inputs)
    shuffled_input_dict = {k: param_dict[k] for k in inputs}
    output = f(**shuffled_input_dict)

    computed_param_dict = output.to_string().values_to_float()

    for k in expected.keys():
        np.testing.assert_allclose(
            computed_param_dict[k], expected[k], err_msg=f"{k} not close to tolerance"
        )
