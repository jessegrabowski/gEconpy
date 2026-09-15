import pytest
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.simplification import reduce_variable_list, simplify_constants, simplify_tryreduce


def test_simplify_tryreduce_substitutes_definition_and_drops_zeroed_equation():
    x = TimeAwareSymbol("x", 0)
    y = TimeAwareSymbol("y", 0)

    reduced_eqs, reduced_vars, eliminated = simplify_tryreduce([x], [x - 1, y - x], [x, y], {x: sp.Integer(1)})

    assert reduced_eqs == [y - 1]
    assert reduced_vars == [y]
    assert eliminated == [x]


def test_simplify_tryreduce_drops_equation_of_variable_appearing_once():
    x, y, z = (TimeAwareSymbol(name, 0) for name in "xyz")

    reduced_eqs, reduced_vars, eliminated = simplify_tryreduce([z], [x - 1, y - x, z - x * y], [x, y, z])

    assert reduced_eqs == [x - 1, y - x]
    assert reduced_vars == [x, y]
    assert eliminated == [z]


def test_simplify_tryreduce_keeps_variable_that_survives_at_a_lag():
    x, y = TimeAwareSymbol("x", 0), TimeAwareSymbol("y", 0)
    eqs = [x - y, y - 2 * x.set_t(-1)]

    reduced_eqs, reduced_vars, eliminated = simplify_tryreduce([x], eqs, [x, y], {x: y})

    assert reduced_eqs == eqs
    assert reduced_vars == [x, y]
    assert eliminated == []


def test_simplify_tryreduce_ignores_variable_without_definition():
    x, y = TimeAwareSymbol("x", 0), TimeAwareSymbol("y", 0)
    eqs = [x - 1, y - x]

    reduced_eqs, reduced_vars, eliminated = simplify_tryreduce([x], eqs, [x, y])

    assert reduced_eqs == eqs
    assert reduced_vars == [x, y]
    assert eliminated == []


def test_simplify_constants_substitutes_constant_at_every_time_index():
    x, y = TimeAwareSymbol("x", 0), TimeAwareSymbol("y", 0)
    eqs = [2 * x - 4, y - x.set_t(-1) - x.set_t(1)]

    reduced_eqs, reduced_vars, eliminated = simplify_constants(eqs, [x, y])

    assert reduced_eqs == [y - 4]
    assert reduced_vars == [y]
    assert eliminated == [x]


def test_simplify_constants_leaves_equation_with_two_variables_alone():
    x, y = TimeAwareSymbol("x", 0), TimeAwareSymbol("y", 0)
    eqs = [x - y, y - x.set_t(-1)]

    reduced_eqs, reduced_vars, eliminated = simplify_constants(eqs, [x, y])

    assert reduced_eqs == eqs
    assert reduced_vars == [x, y]
    assert eliminated == []


def test_reduce_variable_list_sorts_by_name_and_matches_any_time_index():
    a, b, c = (TimeAwareSymbol(name, 0) for name in "abc")

    reduced_vars, eliminated = reduce_variable_list([c.set_t(1) - a.set_t(-1)], [c, b, a])

    assert reduced_vars == [a, c]
    assert eliminated == [b]


@pytest.mark.parametrize(
    "simplify, message",
    [
        (
            lambda eqs, variables: simplify_tryreduce([variables[0]], eqs, variables),
            "Simplification via a tryreduce block was requested but not possible because the system is not well "
            "defined. Found 1 equation but 2 variables",
        ),
        (
            simplify_constants,
            "Removal of constant variables was requested but not possible because the system is not well defined. "
            "Found 1 equation but 2 variables",
        ),
    ],
    ids=["tryreduce", "constants"],
)
def test_non_square_system_warns_and_returns_input_unchanged(simplify, message):
    x, y = TimeAwareSymbol("x", 0), TimeAwareSymbol("y", 0)
    eqs = [x - 1]
    variables = [x, y]

    with pytest.warns(UserWarning, match=message):
        reduced_eqs, reduced_vars, eliminated = simplify(eqs, variables)

    assert reduced_eqs == eqs
    assert reduced_vars == variables
    assert eliminated == []
