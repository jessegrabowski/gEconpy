import numpy as np
import pytest
import sympy as sp

from scipy.optimize import OptimizeResult

from gEconpy.classes.containers import SteadyStateResults
from gEconpy.utilities import flatten_substitution_dict, postprocess_optimizer_res


def test_flatten_substitution_dict_resolves_chained_values():
    a, b, c, d = sp.symbols("a b c d")
    sub_dict = {a: b + 1, b: 2 * c, c: d**2}

    flat = flatten_substitution_dict(sub_dict)

    assert flat == {a: 2 * d**2 + 1, b: 2 * d**2, c: d**2}


def test_flatten_substitution_dict_resolves_independently_of_key_order():
    a, b, c, d = sp.symbols("a b c d")
    forward = flatten_substitution_dict({a: b + 1, b: 2 * c, c: d**2})
    reverse = flatten_substitution_dict({c: d**2, b: 2 * c, a: b + 1})

    assert forward == reverse


def test_flatten_substitution_dict_keeps_plain_values():
    a, b = sp.symbols("a b")
    flat = flatten_substitution_dict({a: b + 1, b: 0.5})

    assert flat == {a: 1.5, b: 0.5}


def test_flatten_substitution_dict_cycle_raises():
    a, b = sp.symbols("a b")
    with pytest.raises(ValueError, match="Substitution dictionary has a cycle"):
        flatten_substitution_dict({a: b + 1, b: a - 1})


def _optimizer_check(success, resid_scale):
    res = OptimizeResult(success=success, message="stub")
    res_dict = SteadyStateResults({"x_ss": 1.0})

    def f_resid(x_ss):
        return np.array([resid_scale * x_ss])

    def f_grad(x_ss):
        return np.array([[resid_scale * x_ss]])

    return postprocess_optimizer_res(res, res_dict, f_resid, f_grad, tol=1e-6, verbose=False)


@pytest.mark.parametrize(
    "optimizer_success, resid_scale, expected",
    [
        (True, 1e-9, True),
        (False, 1e-9, True),
        (True, 1e-3, False),
        (False, 1e-3, False),
    ],
    ids=["both_pass", "numeric_overrides_optimizer", "optimizer_flag_alone_insufficient", "both_fail"],
)
def test_postprocess_optimizer_res_success_requires_numeric_check(optimizer_success, resid_scale, expected):
    res_dict = _optimizer_check(success=optimizer_success, resid_scale=resid_scale)
    assert res_dict.success == expected
