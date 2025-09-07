import pytest
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.simplification import simplify_constants, simplify_tryreduce


def test_simplify_tryreduce_basic():
    x = TimeAwareSymbol("x", 0)
    y = TimeAwareSymbol("y", 0)

    # x - 1 implies that x = 1, so x can be eliminated
    eqs = [x - 1, y - x]
    variables = [x, y]
    try_reduce_vars = [x]
    tryreduce_sub_dict = {x: 1}

    reduced_eqs, reduced_vars, eliminated = simplify_tryreduce(try_reduce_vars, eqs, variables, tryreduce_sub_dict)

    assert all(isinstance(eq, sp.Expr) for eq in reduced_eqs)
    assert y in reduced_vars
    assert x in eliminated


def test_simplify_constants_basic():
    x = TimeAwareSymbol("x", 0)
    y = TimeAwareSymbol("y", 0)

    # x - 1 implies that x = 1, so x can be eliminated as a constant
    eqs = [x - 1, y - x]
    variables = [x, y]

    reduced_eqs, reduced_vars, eliminated = simplify_constants(eqs, variables)
    assert all(isinstance(eq, sp.Expr) for eq in reduced_eqs)
    assert x not in reduced_vars
    assert x in eliminated


def test_simplify_tryreduce_non_square_warns():
    x = TimeAwareSymbol("x", 0)
    y = TimeAwareSymbol("y", 0)

    eqs = [x - 1]
    variables = [x, y]
    try_reduce_vars = [x]

    with pytest.warns(
        UserWarning,
        match="Simplification via a tryreduce block was requested but not possible because "
        "the system is not well defined.",
    ):
        simplify_tryreduce(try_reduce_vars, eqs, variables)


def test_simplify_constants_non_square_warns():
    x = TimeAwareSymbol("x", 0)
    y = TimeAwareSymbol("y", 0)
    eqs = [x - 1]
    variables = [x, y]

    with pytest.warns(
        UserWarning,
        match="Removal of constant variables was requested but not possible because the system is not well defined.",
    ):
        simplify_constants(eqs, variables)
