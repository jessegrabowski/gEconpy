from collections.abc import Collection, Sequence
from dataclasses import dataclass

import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser.ast import SteadyStateSolverCall
from gEconpy.utilities import (
    _rebuild_unevaluated,
    _xreplace_unevaluated,
    eq_to_ss,
    flatten_substitution_dict,
    substitute_unevaluated,
)


@dataclass(frozen=True)
class SteadyStateSolve:
    """
    A steady-state variable the steady state block solves for numerically with ``root`` or ``minimize``.

    Attributes
    ----------
    call : SteadyStateSolverCall
        The call as written, which carries the method and its starting point.
    expression : sympy expression
        The call's expression, in which the solved variable and other steady-state variables appear as symbols.
    """

    call: SteadyStateSolverCall
    expression: sp.Expr


def resubstitute_steady_state(relationships: Sequence[sp.Eq]) -> list[sp.Eq]:
    """
    Rewrite each steady-state relationship in terms of the other steady-state variables, for printing.

    A relationship the model holds has every other steady-state value inlined into it, so ``K_ss`` repeats the
    whole expression for ``r_ss``. Wherever a subexpression is exactly another relationship's right side, it is
    replaced by that relationship's variable. Matching is structural and top-down, so the largest match wins and
    no new names are introduced. A right side that is a single number or symbol is never a replacement target,
    because a constant such as ``A_ss = 1`` would otherwise rename every literal one.

    The trees are rebuilt unevaluated: evaluating an inlined steady-state tree costs time exponential in its
    depth.

    Parameters
    ----------
    relationships : sequence of Eq
        Steady-state variables equal to their solutions, as ``Model.steady_state_relationships`` holds them.

    Returns
    -------
    relationships : list of Eq
        The same relationships, in the same order, with every other variable's solution folded back in.
    """
    targets = {eq.rhs: eq.lhs for eq in relationships if not (eq.rhs.is_Number or eq.rhs.is_Symbol)}
    cache: dict[sp.Basic, sp.Basic] = {}

    resubstituted = []
    for eq in relationships:
        others = {rhs: lhs for rhs, lhs in targets.items() if lhs != eq.lhs}
        rhs = _drop_identity_terms(_xreplace_unevaluated(eq.rhs, others), cache)
        resubstituted.append(sp.Eq(eq.lhs, rhs, evaluate=False))
    return resubstituted


def _drop_identity_terms(expr: sp.Basic, cache: dict[sp.Basic, sp.Basic]) -> sp.Basic:
    """
    Remove the ones and zeros substitution leaves in an unevaluated tree, and print an integral float as an integer.

    A one is a factor of a product or the base of a power, and a zero is a term of a sum.

    The cache is keyed on subtrees, because an inlined tree shares them and a plain recursion revisits each share.
    """
    if expr in cache:
        return cache[expr]

    if isinstance(expr, sp.Float) and float(expr).is_integer():
        cleaned = sp.Integer(int(expr))
    elif expr.is_Atom or not expr.args:
        cleaned = expr
    else:
        args = [_drop_identity_terms(arg, cache) for arg in expr.args]
        identity = {sp.Mul: sp.Integer(1), sp.Add: sp.Integer(0)}.get(type(expr))
        if identity is not None:
            args = [arg for arg in args if arg != identity] or [identity]
        if isinstance(expr, sp.Pow) and args[0] == 1:
            cleaned = sp.Integer(1)
        elif identity is not None and len(args) == 1:
            cleaned = args[0]
        else:
            cleaned = _rebuild_unevaluated(expr, args)

    cache[expr] = cleaned
    return cleaned


def make_steady_state_shock_dict(shocks: list[TimeAwareSymbol]) -> SymbolDictionary:
    """
    Build a substitution dictionary setting every shock to zero in the steady state.

    Parameters
    ----------
    shocks : list of TimeAwareSymbol
        Model shocks.

    Returns
    -------
    shock_dict : SymbolDictionary
        Mapping from each steady-state shock symbol to 0.0.
    """
    return SymbolDictionary.fromkeys(shocks, 0.0).to_ss()


def system_to_steady_state(system: list[sp.Expr], shocks: list[TimeAwareSymbol]) -> list[sp.Expr]:
    """
    Rewrite a dynamic system of equations in steady-state form.

    Parameters
    ----------
    system : list of sympy expression
        Model equations in time-indexed variables.
    shocks : list of TimeAwareSymbol
        Model shocks, which are set to zero.

    Returns
    -------
    system : list of sympy expression
        Simplified steady-state equations.
    """
    # One steady-state equation per dynamic equation, in the same order, so an equation id indexes both lists.
    shock_dict = make_steady_state_shock_dict(shocks)
    return [eq_to_ss(eq).subs(shock_dict).simplify() for eq in system]


def simplify_provided_ss_equations(
    ss_solution_dict: SymbolDictionary, variables: list[TimeAwareSymbol]
) -> SymbolDictionary:
    """
    Substitute intermediate definitions out of the user-provided steady-state equations.

    Parameters
    ----------
    ss_solution_dict : SymbolDictionary
        Steady-state equations from the ``STEADY_STATE`` block.
    variables : list of TimeAwareSymbol
        Model variables.

    Returns
    -------
    ss_solution_dict : SymbolDictionary
        Steady-state equations keyed by model variables only.
    """
    if not ss_solution_dict:
        return SymbolDictionary()

    ss_variables = [variable.to_ss() for variable in variables]
    ss_dict_sympy = ss_solution_dict.to_sympy()

    intermediates = {key: value for key, value in ss_dict_sympy.items() if key not in ss_variables}
    if not intermediates:
        return ss_solution_dict

    simplified = SymbolDictionary({key: value for key, value in ss_dict_sympy.items() if key in ss_variables})
    flat_intermediates = flatten_substitution_dict(intermediates, evaluate=False)
    for variable, eq in simplified.items():
        if isinstance(eq, sp.Basic):
            simplified[variable] = substitute_unevaluated(eq, flat_intermediates)

    return simplified


def propagate_steady_state_through_identities(
    ss_solution_dict: SymbolDictionary,
    steady_state_equations: list[sp.Expr],
    variables: list[TimeAwareSymbol],
    solved_numerically: Collection[sp.Symbol] = (),
    max_iterations: int = 100,
) -> SymbolDictionary:
    """
    Extend the user-provided steady state by solving equations with a single remaining unknown.

    Each pass substitutes the known values into every equation and solves those left with exactly one unknown. Passes
    repeat until one makes no progress. A solution is accepted only if it is unique and simple: no conditionals, no
    unevaluated integrals or derivatives, no complex numbers, and limited nesting depth.

    Parameters
    ----------
    ss_solution_dict : SymbolDictionary
        User-provided steady-state values.
    steady_state_equations : list of sympy expression
        Model equations in steady-state residual form, each equal to zero.
    variables : list of TimeAwareSymbol
        Model variables.
    solved_numerically : iterable of sympy Symbol, optional
        Steady-state variables solved numerically elsewhere, such as by a ``root`` call in the steady state block.
        They are never solved for, so an equation counts them as known. Default is none.
    max_iterations : int, optional
        Maximum number of passes over the equation system. Default is 100.

    Returns
    -------
    ss_solution_dict : SymbolDictionary
        The provided values plus every value that could be inferred.
    """
    ss_variables = {variable.to_ss() for variable in variables}
    known = ss_solution_dict.to_sympy().copy() if ss_solution_dict else {}
    never_solved = set(solved_numerically)

    for _ in range(max_iterations):
        progress = False

        for eq in steady_state_equations:
            if not isinstance(eq, sp.Basic):
                continue

            unknowns = (eq.free_symbols & ss_variables) - set(known.keys()) - never_solved
            if len(unknowns) != 1:
                continue

            unknown = next(iter(unknowns))
            solution = _try_solve_for_unknown(eq, unknown, known, ss_variables)
            if solution is None:
                continue

            known[unknown] = float(solution) if solution.is_number else solution
            progress = True

        if not progress:
            break

    return SymbolDictionary(known)


def _try_solve_for_unknown(
    eq: sp.Expr,
    unknown: sp.Symbol,
    known_values: dict[sp.Symbol, sp.Expr],
    ss_variables: set[sp.Symbol],
) -> sp.Expr | None:
    """Solve ``eq`` for ``unknown`` after substituting the known values, or return None if no simple unique solution."""
    eq_substituted = eq.subs(known_values)

    remaining_ss_vars = eq_substituted.free_symbols & ss_variables
    if remaining_ss_vars != {unknown}:
        return None

    try:
        solutions = sp.solve(eq_substituted, unknown, dict=False)
    except (NotImplementedError, ValueError, TypeError):
        return None

    if len(solutions) != 1:
        return None

    solution = solutions[0]
    return solution if _solution_is_simple(solution) else None


def _solution_is_simple(expr: sp.Expr, max_nesting_depth: int = 5) -> bool:
    """Reject solutions with conditionals, unevaluated calculus, complex numbers, or deep nesting."""
    if expr.has(sp.Piecewise, sp.Integral, sp.Derivative, sp.I):
        return False

    def depth(node: sp.Basic, current: int = 0) -> int:
        return current if not node.args else max(depth(arg, current + 1) for arg in node.args)

    return depth(expr) <= max_nesting_depth
