import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

import gEconpy.pytensorf.optimize

numba = pytest.importorskip("numba")

# Any object-mode fallback warns, so this fails a test whose dispatch falls back.
pytestmark = pytest.mark.filterwarnings("error")

# The root of x + 2 * a * cos(x) near zero at a = 1.
ROOT = -1.02986653
METHODS = [
    ("newton", {"jac": True}),
    ("halley", {"jac": True, "hess": True}),
    ("secant", {}),
    ("brentq", {"optimizer_kwargs": {"bracket": (-2.0, 0.0)}}),
    ("bisect", {"optimizer_kwargs": {"bracket": (-2.0, 0.0)}}),
]


def _root_function(method, options, mode, gradient=False):
    x = pt.dscalar("x")
    a = pt.dscalar("a")
    solution, success = pt.optimize.root_scalar(x + 2 * a * pt.cos(x), x, method=method, **options)
    outputs = [pytensor.grad(solution, a)] if gradient else [solution, success]
    return pytensor.function([x, a], outputs, mode=mode)


@pytest.mark.parametrize("method, options", METHODS, ids=[method for method, _ in METHODS])
def test_numba_solves_without_object_mode(method, options):
    root, success = _root_function(method, options, mode="NUMBA")(0.0, 1.0)

    assert success
    np.testing.assert_allclose(root, ROOT, atol=1e-8)


@pytest.mark.parametrize("method, options", METHODS, ids=[method for method, _ in METHODS])
def test_numba_gradient_is_the_implicit_derivative(method, options):
    """At the root of f(x, a) = x + 2 a cos(x), dx/da = -f_a / f_x = -2 cos(x) / (1 - 2 a sin(x))."""
    gradient = _root_function(method, options, mode="NUMBA", gradient=True)(0.0, 1.0)

    np.testing.assert_allclose(gradient, -2 * np.cos(ROOT) / (1 - 2 * np.sin(ROOT)), rtol=1e-6)


@pytest.mark.parametrize("method, options", METHODS, ids=[method for method, _ in METHODS])
def test_a_one_iteration_cap_stops_the_solve_before_it_converges(method, options):
    capped = {**options, "optimizer_kwargs": {**options.get("optimizer_kwargs", {}), "maxiter": 1}}

    _root, success = _root_function(method, capped, mode="NUMBA")(0.0, 1.0)

    assert not success


@pytest.mark.parametrize("method", ["brentq", "bisect"])
@pytest.mark.parametrize(
    "equation, bracket",
    [
        (lambda x: x + 2 * pt.cos(x), (0.0, 1.0)),
        (lambda x: pt.log(x) - 0.5, (-1.0, 3.0)),
    ],
    ids=["no_sign_change", "nan_at_an_end"],
)
def test_an_invalid_bracket_does_not_converge(method, equation, bracket):
    x = pt.dscalar("x")
    solution, success = pt.optimize.root_scalar(equation(x), x, method=method, optimizer_kwargs={"bracket": bracket})

    root, converged = pytensor.function([x], [solution, success], mode="NUMBA")(0.0)

    assert not converged
    assert np.isnan(root)


def test_the_root_keeps_the_variable_dtype():
    x = pt.fscalar("x")
    a = pt.fscalar("a")
    solution, _ = pt.optimize.root_scalar(x + 2 * a * pt.cos(x), x, method="secant")

    root = pytensor.function([x, a], solution, mode="NUMBA")(np.float32(0.0), np.float32(1.0))

    assert root.dtype == np.float32
    np.testing.assert_allclose(root, ROOT, rtol=1e-6)


def test_an_option_without_a_numba_implementation_falls_back_to_object_mode():
    with pytest.warns(UserWarning, match="Numba will use object mode"):
        function = _root_function("newton", {"jac": True, "optimizer_kwargs": {"rtol": 1e-12}}, mode="NUMBA")

    root, success = function(0.0, 1.0)

    assert success
    np.testing.assert_allclose(root, ROOT, atol=1e-8)
