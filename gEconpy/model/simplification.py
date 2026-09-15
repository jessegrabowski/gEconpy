import warnings

import numpy as np
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.utilities import (
    expand_subs_for_all_times,
    make_all_var_time_combos,
    substitute_all_equations,
)

# An equation that pins a variable to a constant has at most three atoms, as in ``2*x - 1`` -> {2, x, -1}.
_MAX_ATOMS_IN_CONSTANT_EQUATION = 3


def simplify_tryreduce(
    try_reduce_vars: list[TimeAwareSymbol],
    equations: list[sp.Expr],
    variables: list[TimeAwareSymbol],
    tryreduce_sub_dict: dict[TimeAwareSymbol, sp.Expr] | None = None,
) -> tuple[list[sp.Expr], list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    """
    Eliminate the variables listed in the ``tryreduce`` block of a GCN file where doing so is safe.

    A variable is eliminated in one of two ways. If it appears in exactly one equation, that equation is dropped, since
    no other equation depends on the variable. Otherwise its defining equation from ``tryreduce_sub_dict`` is
    substituted into the system, and the substitution is kept only if it zeroes exactly one equation and removes the
    variable from every other.

    Parameters
    ----------
    try_reduce_vars : list of TimeAwareSymbol
        Variables to try to eliminate.
    equations : list of sympy expression
        Model equations.
    variables : list of TimeAwareSymbol
        All variables in the system.
    tryreduce_sub_dict : dict, optional
        Mapping from each variable to the expression that defines it. Default is None, meaning no substitutions are
        attempted.

    Returns
    -------
    reduced_equations : list of sympy expression
        Equations that remain after elimination.
    reduced_variables : list of TimeAwareSymbol
        Variables that remain in the system.
    eliminated_vars : list of TimeAwareSymbol
        Variables that were removed.
    """
    n_equations = len(equations)
    n_variables = len(variables)
    if not _check_system_is_square("Simplification via a tryreduce block", n_equations, n_variables):
        return equations, variables, []

    if tryreduce_sub_dict is None:
        tryreduce_sub_dict = {}

    occurrence_matrix = np.zeros((n_equations, n_variables))
    combo_to_col = {}
    for j, var in enumerate(variables):
        for sym in make_all_var_time_combos([var]):
            combo_to_col[sym] = j

    for i, eq in enumerate(equations):
        cols = {combo_to_col[sym] for sym in eq.atoms(sp.Symbol) if sym in combo_to_col}
        if cols:
            occurrence_matrix[i, list(cols)] += 1

    isolated_variables = np.array(variables)[occurrence_matrix.sum(axis=0) == 1]
    to_remove = set(isolated_variables).intersection(set(try_reduce_vars))
    reduced_equations = [eq for eq in equations if not any(var in eq.atoms() for var in to_remove)]

    for reduction_variable in try_reduce_vars:
        if reduction_variable not in tryreduce_sub_dict:
            continue

        sub_dict = {reduction_variable: tryreduce_sub_dict[reduction_variable]}
        candidate = [eq.simplify() for eq in substitute_all_equations(reduced_equations, sub_dict)]

        all_time_indices = make_all_var_time_combos([reduction_variable])
        variable_remains = any(sym in eq.atoms() for eq in candidate for sym in all_time_indices)
        if candidate.count(0) == 1 and not variable_remains:
            reduced_equations = [eq for eq in candidate if eq != 0]

    reduced_variables, eliminated_vars = reduce_variable_list(reduced_equations, variables)
    return reduced_equations, reduced_variables, eliminated_vars


def simplify_constants(
    equations: list[sp.Expr], variables: list[TimeAwareSymbol]
) -> tuple[list[sp.Expr], list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    """
    Substitute away variables that an equation pins to a constant.

    Typical cases are ``P[] = 1``, which makes the price level the numeraire, and ``B[] = 0``, which puts bonds in
    zero net supply. Run this after the first-order conditions are derived, so that the variable is replaced by its
    value everywhere it appears.

    Parameters
    ----------
    equations : list of sympy expression
        Model equations.
    variables : list of TimeAwareSymbol
        All variables in the system.

    Returns
    -------
    reduced_equations : list of sympy expression
        Equations that remain after substitution.
    reduced_variables : list of TimeAwareSymbol
        Variables that remain in the system.
    eliminated_vars : list of TimeAwareSymbol
        Variables that were removed.
    """
    if not _check_system_is_square("Removal of constant variables", len(equations), len(variables)):
        return equations, variables, []

    reduce_dict = {}
    for eq in equations:
        if len(eq.atoms()) > _MAX_ATOMS_IN_CONSTANT_EQUATION:
            continue

        equation_variables = list(eq.atoms(TimeAwareSymbol))
        if len(equation_variables) != 1:
            continue

        solution = sp.solve(eq, equation_variables[0], dict=True)[0]
        reduce_dict.update(expand_subs_for_all_times(solution))

    reduced_equations = [eq for eq in substitute_all_equations(equations, reduce_dict) if eq != 0]
    reduced_variables, eliminated_vars = reduce_variable_list(reduced_equations, variables)

    return reduced_equations, reduced_variables, eliminated_vars


def reduce_variable_list(
    equations: list[sp.Expr], variables: list[TimeAwareSymbol]
) -> tuple[list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    """
    Split a variable list into the variables that still appear in a system of equations and those that do not.

    Parameters
    ----------
    equations : list of sympy expression
        Equations to scan for variables.
    variables : list of TimeAwareSymbol
        Variables to classify, at time index t.

    Returns
    -------
    reduced_variables : list of TimeAwareSymbol
        Variables that appear in ``equations``, sorted by name.
    eliminated_vars : list of TimeAwareSymbol
        Variables that do not appear in ``equations``, sorted by name.
    """
    variable_set = set(variables)
    present = {atom.set_t(0) for eq in equations for atom in eq.atoms(TimeAwareSymbol) if atom.set_t(0) in variable_set}

    reduced_variables = sorted(present, key=lambda x: x.name)
    eliminated_vars = sorted(variable_set - present, key=lambda x: x.name)

    return reduced_variables, eliminated_vars


def _check_system_is_square(msg: str, n_equations: int, n_variables: int) -> bool:
    if n_equations == n_variables:
        return True

    warnings.warn(
        f"{msg} was requested but not possible because the system is not well defined. "
        f"Found {n_equations} equation{'s' if n_equations > 1 else ''} but {n_variables} variable"
        f"{'s' if n_variables > 1 else ''}",
        stacklevel=2,
    )
    return False
