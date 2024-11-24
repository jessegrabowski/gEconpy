from warnings import warn

import numpy as np
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.utilities import (
    expand_subs_for_all_times,
    is_variable,
    make_all_var_time_combos,
    substitute_all_equations,
)


def _check_system_is_square(msg: str, n_equations: int, n_variables: int) -> bool:
    if n_equations != n_variables:
        warn(
            f'{msg} was requested but not possible because the system is not well defined. '
            f'Found {n_equations} equation{"s" if n_equations > 1 else ""} but {n_variables} variable'
            f'{"s" if n_variables > 1 else ""}'
        )
        return False
    return True


def reduce_variable_list(equations, variables):
    reduced_variables = {
        atom.set_t(0)
        for eq in equations
        for atom in eq.atoms()
        if is_variable(atom) and atom.set_t(0) in variables
    }

    reduced_variables = sorted(list(reduced_variables), key=lambda x: x.name)
    eliminated_vars = sorted(
        list(set(variables) - set(reduced_variables)), key=lambda x: x.name
    )

    return reduced_variables, eliminated_vars


def simplify_tryreduce(
    try_reduce_vars: list[TimeAwareSymbol],
    equations: list[sp.Expr],
    variables: list[TimeAwareSymbol],
    tryreduce_sub_dict: dict[TimeAwareSymbol, sp.Expr] | None = None,
) -> tuple[list[sp.Expr], list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    """
    Attempt to reduce the number of equations in the system by removing equations requested in the `tryreduce`
    block of the GCN file. Equations are considered safe to remove if they are "self-contained" that is, if
    no other variables depend on their values.

    Returns
    -------
    list
        The names of the variables that were removed. If reduction was not possible, None is returned.
    """
    n_equations = len(equations)
    n_variables = len(variables)
    if not _check_system_is_square(
        "Simplification via a tryreduce block", n_equations, n_variables
    ):
        return equations, variables, []
    if tryreduce_sub_dict is None:
        tryreduce_sub_dict = {}

    occurrence_matrix = np.zeros((n_variables, n_variables))
    reduced_equations = []

    for i, eq in enumerate(equations):
        for j, var in enumerate(variables):
            if any([x in eq.atoms() for x in make_all_var_time_combos([var])]):
                occurrence_matrix[i, j] += 1

    # Columns with a sum of 1 are variables that appear only in a single equations; these equations can be deleted
    # without consequence w.r.t solving the system, with no further checking required.
    isolated_variables = np.array(variables)[occurrence_matrix.sum(axis=0) == 1]
    to_remove = set(isolated_variables).intersection(set(try_reduce_vars))

    for eq in equations:
        if not any([var in eq.atoms() for var in to_remove]):
            reduced_equations.append(eq)

    # Next use the user-supplied equations to reduce the system further, seeking to eliminate variables via direct
    # substitution.
    for reduction_variable in try_reduce_vars:
        if reduction_variable not in tryreduce_sub_dict:
            continue
        sub_dict = {reduction_variable: tryreduce_sub_dict[reduction_variable]}
        reduction_candidate = substitute_all_equations(reduced_equations, sub_dict)
        reduction_candidate = [eq.simplify() for eq in reduction_candidate]

        # To be a valid reduction, there should be exactly one zero in reduction_candidates, and the reduced variable
        # should no longer appear in the system.
        if reduction_candidate.count(0) == 1 and not any(
            [
                x in eq.atoms()
                for eq in reduction_candidate
                for x in make_all_var_time_combos([reduction_variable])
            ]
        ):
            reduced_equations = [eq for eq in reduction_candidate if eq != 0]

    reduced_variables, eliminated_vars = reduce_variable_list(
        reduced_equations, variables
    )
    return reduced_equations, reduced_variables, eliminated_vars


def simplify_constants(
    equations: list[sp.Expr], variables: list[TimeAwareSymbol]
) -> tuple[list[sp.Expr], list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    """
    Simplify the system by removing variables that are deterministically defined as a known value. Common examples
    include P[] = 1, setting the price level of the economy as the numeraire, or B[] = 0, putting the bond market
    in net-zero supply.

    In these cases, the variable can be replaced by the deterministic value after all FoC
    have been computed.

    Returns
    -------
    eliminated_vars : List[str]
        The names of the variables that were removed.
    """
    n_equations = len(equations)
    n_variables = len(variables)

    if not _check_system_is_square(
        "Removal of constant variables", n_equations, n_variables
    ):
        return equations, variables, []

    reduce_dict = {}

    for eq in equations:
        if len(eq.atoms()) < 4:
            var = [x for x in eq.atoms() if is_variable(x)]
            if len(var) != 1:
                continue
            var = var[0]
            sub_dict = expand_subs_for_all_times(sp.solve(eq, var, dict=True)[0])
            reduce_dict.update(sub_dict)

    reduced_equations = substitute_all_equations(equations, reduce_dict)
    reduced_equations = [eq for eq in reduced_equations if eq != 0]

    reduced_variables, eliminated_vars = reduce_variable_list(
        reduced_equations, variables
    )

    return reduced_equations, reduced_variables, eliminated_vars
