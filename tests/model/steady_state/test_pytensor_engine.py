import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import sympy as sp

from numpy.testing import assert_allclose
from pymc.distributions.transforms import Interval, log, logodds
from pytensor.graph.traversal import ancestors

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import make_cache_key, sympy_to_pytensor
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.model.steady_state.pytensor_engine import (
    build_steady_state_system,
    compile_known_ss,
    infer_variable_bounds,
    infer_variable_transform,
    known_steady_state_graphs,
    pt_error_from_resid,
    steady_state_transforms,
    transform_steady_state_system,
)
from tests._resources.cache_compiled_models import load_and_cache_model


@pytest.mark.parametrize(
    "error_function, expected",
    [("squared", 14.0), ("mean_squared", 14.0 / 3), ("abs", 6.0), ("l2-norm", np.sqrt(14.0))],
    ids=str,
)
def test_pt_error_from_resid(error_function, expected):
    resid = pt.dvector("resid")
    f = pytensor.function([resid], pt_error_from_resid(resid, error_function), mode="FAST_COMPILE")

    assert_allclose(f(np.array([1.0, -2.0, 3.0])), expected)


def test_pt_error_from_resid_rejects_unknown_function():
    with pytest.raises(NotImplementedError, match="Error function huber not implemented"):
        pt_error_from_resid(pt.dvector("resid"), "huber")


def test_compile_known_ss_symbolic_keys_only_known_variables():
    model = load_and_cache_model("rbc_2_block_partial_ss.gcn")
    _, cache = compile_param_dict_func(model._param_dict, model._deterministic_dict, return_symbolic=True)
    parameters = list(model._param_dict.to_sympy().keys()) + list(model._deterministic_dict.to_sympy().keys())

    mapping, _ = compile_known_ss(
        model._ss_solution_dict, model.variables, parameters, cache=cache, return_symbolic=True
    )

    known_names = [symbol.name for symbol in model._ss_solution_dict.to_sympy()]
    assert [node.name for node in mapping] == known_names


class TestKnownSteadyStateGraphs:
    alpha = sp.Symbol("alpha", positive=True)
    a_ss, b_ss = sp.symbols("a_ss b_ss", positive=True)
    definitions = {b_ss: a_ss ** (1 / (1 - alpha)) + a_ss, a_ss: 2 * alpha}
    flat = {a_ss: 2 * alpha, b_ss: (2 * alpha) ** (1 / (1 - alpha)) + 2 * alpha}

    def _evaluate(self, known, definitions):
        cache = {}
        (alpha_pt,), _, cache = sympy_to_pytensor([self.alpha], [], cache)
        graphs = known_steady_state_graphs(known, definitions, cache)
        return graphs, pytensor.function([alpha_pt], graphs)(0.3)

    def test_matches_inlined_values_and_reuses_definition_graphs(self):
        graphs, values = self._evaluate(self.flat, self.definitions)
        _, expected = self._evaluate(self.flat, None)

        assert_allclose(values, expected)
        a_graph, b_graph = graphs
        assert a_graph in ancestors([b_graph])

    def test_numeric_value_overrides_only_its_own_entry(self):
        _, values = self._evaluate({self.a_ss: 5.0, self.b_ss: self.flat[self.b_ss]}, self.definitions)

        assert values[0] == 5.0
        assert_allclose(values[1], 0.6 ** (1 / 0.7) + 0.6)

    def test_value_without_a_block_entry_is_converted_directly(self):
        c_ss = sp.Symbol("c_ss", positive=True)

        _, values = self._evaluate({c_ss: 3 * self.alpha}, self.definitions)

        assert_allclose(values, [0.9])

    def test_shared_cache_keeps_steady_state_symbol_as_input(self):
        cache = {}
        sympy_to_pytensor([self.alpha, self.a_ss], [], cache)
        key = make_cache_key(self.a_ss.name, type(self.a_ss))
        input_node = cache[key]

        known_steady_state_graphs(self.flat, self.definitions, cache)

        assert cache[key] is input_node


@pytest.mark.parametrize(
    "assumptions, user_bound, expected_type",
    [
        ({"positive": True}, (0.0, 1.0), Interval),
        ({"unit_interval": True, "positive": True}, None, type(logodds)),
        ({"positive": True}, None, type(log)),
        ({"negative": True}, None, Interval),
        ({}, None, type(None)),
    ],
    ids=["explicit_bound_wins", "unit_interval", "positive", "negative", "unconstrained"],
)
def test_infer_variable_transform_waterfall(assumptions, user_bound, expected_type):
    variable = sp.Symbol("x", **assumptions)
    transform = infer_variable_transform(variable, user_bound=user_bound)
    assert isinstance(transform, expected_type)


@pytest.mark.parametrize(
    "assumptions, expected",
    [({"positive": True}, (1e-8, None)), ({"negative": True}, (None, -1e-8)), ({}, (None, None))],
    ids=["positive", "negative", "unconstrained"],
)
def test_infer_variable_bounds(assumptions, expected):
    assert infer_variable_bounds(TimeAwareSymbol("x", 0, **assumptions)) == expected


@pytest.mark.parametrize(
    "declared, user, expected_midpoint",
    [
        ({"x_t": (0.0, 1.0)}, None, 0.5),
        ({"x_t": (0.0, 1.0)}, {"x_ss": (2.0, 4.0)}, 3.0),
        ({}, {"x_ss": (2.0, 4.0)}, 3.0),
        ({"x_t": (0.0, 1.0)}, {"x": (2.0, 4.0)}, 0.5),
    ],
    ids=["declared_bound", "caller_bound_wins", "caller_bound_only", "caller_bound_needs_the_steady_state_name"],
)
def test_steady_state_transforms_precedence(declared, user, expected_midpoint):
    """A declared bound is keyed by the variable's time-t name and a caller bound by the steady-state name."""
    variable = TimeAwareSymbol("x", "ss")
    transform = steady_state_transforms([variable], declared, user)[0]

    assert float(transform.backward(0.0).eval()) == pytest.approx(expected_midpoint)


def test_declared_bounds_find_a_parameter_by_its_own_name():
    """A calibrated parameter is a plain Symbol with no base name, so the declared bound is keyed by name."""
    transform = steady_state_transforms([sp.Symbol("psi")], {"psi": (1.0, 3.0)}, {})[0]

    assert float(transform.backward(0.0).eval()) == pytest.approx(2.0)


def test_steady_state_transforms_fall_back_to_assumptions():
    variable = TimeAwareSymbol("x", "ss", positive=True)
    assert isinstance(steady_state_transforms([variable], {}, {})[0], type(log))


def test_build_steady_state_system_splits_rows_and_lets_fixed_values_win():
    x, y, z, p = pt.dscalars("x_ss", "y_ss", "z_ss", "p")
    equations = [x - p, y - 2 * x, z - y, z - 3.0]
    known = {x: 2.0 * p, z: p + 1.0}

    system = build_steady_state_system(equations, [x, y, z], known, fixed=[z])

    assert system.unknowns == [y]
    assert system.check_indices == [0, 3]
    assert len(system.residuals) == 2
    assert system.steady_state[1] is y
    assert system.steady_state[2] is z

    values = {p: 0.5, y: 2.0, z: 3.0}
    assert_allclose([row.eval(values, on_unused_input="ignore") for row in system.checks], [0.5, 0.0])
    assert_allclose([row.eval(values, on_unused_input="ignore") for row in system.residuals], [0.0, 1.0])
    assert_allclose(system.steady_state[0].eval({p: 0.5}), 1.0)


def test_transform_steady_state_system_round_trips_and_preserves_residuals():
    x, y = pt.dscalars("x", "y")
    equations = [x - 2.0, y + x]

    transformed, y_nodes, to_unconstrained, to_constrained = transform_steady_state_system(
        equations, [x, y], [log, None]
    )

    f_original = pytensor.function([x, y], equations, mode="FAST_COMPILE")
    f_transformed = pytensor.function(y_nodes, transformed, mode="FAST_COMPILE")

    constrained_point = np.array([3.0, -1.5])
    unconstrained_point = to_unconstrained(constrained_point)

    assert_allclose(unconstrained_point, [np.log(3.0), -1.5])
    assert_allclose(to_constrained(unconstrained_point), constrained_point)
    assert_allclose(f_transformed(*unconstrained_point), f_original(*constrained_point))
