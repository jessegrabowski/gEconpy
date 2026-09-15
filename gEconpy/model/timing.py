import re

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol

VALID_TIME_INDICES = frozenset({-1, 0, 1})


def natural_sort_key(symbol: TimeAwareSymbol) -> list[str | int]:
    """
    Sort key that orders numeric suffixes by value, so ``x1, x2, x10`` sort in that order.

    Parameters
    ----------
    symbol : TimeAwareSymbol
        Symbol whose ``base_name`` is the sort key.

    Returns
    -------
    key : list of str and int
        The base name split into alternating text and integer parts.
    """
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", symbol.base_name)]


def collect_time_aware_atoms(equations: list[sp.Expr]) -> set[TimeAwareSymbol]:
    """
    Collect every :class:`~gEconpy.classes.time_aware_symbol.TimeAwareSymbol` atom in a list of expressions.

    Parameters
    ----------
    equations : list of Expr
        Sympy expressions to scan.

    Returns
    -------
    atoms : set of TimeAwareSymbol
        The union of the time-aware atoms of every expression.
    """
    return set().union(*(equation.atoms(TimeAwareSymbol) for equation in equations))


def classify_variables_by_timing(
    equations: list[sp.Expr],
    shock_names: list[str],
) -> tuple[list[TimeAwareSymbol], list[TimeAwareSymbol], list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    """
    Group the variables and shocks appearing in the equations by time index.

    Only symbols that appear at a given time index are listed under it. Every list is sorted with
    :func:`natural_sort_key`.

    Parameters
    ----------
    equations : list of Expr
        Model equations as sympy expressions.
    shock_names : list of str
        Base names of the exogenous shocks.

    Returns
    -------
    vars_tm1 : list of TimeAwareSymbol
        Variables at t-1.
    vars_t : list of TimeAwareSymbol
        Variables at t.
    vars_tp1 : list of TimeAwareSymbol
        Variables at t+1.
    shocks : list of TimeAwareSymbol
        Shocks at any time index.
    """
    all_atoms = collect_time_aware_atoms(equations)
    shock_name_set = set(shock_names)

    invalid = {x for x in all_atoms if x.time_index not in VALID_TIME_INDICES}
    if invalid:
        bad = ", ".join(f"{x.name} (t={x.time_index})" for x in sorted(invalid, key=natural_sort_key))
        raise ValueError(
            f"Equations contain variables at unexpected time indices. Expected only t-1, t, or t+1, "
            f"found: {bad}. Equations should be normalized before classification."
        )

    endogenous_by_name_and_time: dict[tuple[str, int | str], TimeAwareSymbol] = {
        (x.base_name, x.time_index): x for x in all_atoms if x.base_name not in shock_name_set
    }

    def vars_at_time(t: int) -> list[TimeAwareSymbol]:
        symbols_at_t = [symbol for (_, time_index), symbol in endogenous_by_name_and_time.items() if time_index == t]
        return sorted(symbols_at_t, key=natural_sort_key)

    shocks = sorted([x for x in all_atoms if x.base_name in shock_name_set], key=natural_sort_key)
    return vars_at_time(-1), vars_at_time(0), vars_at_time(1), shocks


def make_all_variable_time_combinations(
    variables: list[TimeAwareSymbol],
) -> tuple[list[TimeAwareSymbol], list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    """
    Produce the t-1, t, and t+1 variants of every variable.

    Each variable is first normalized to t. Duplicate base names are dropped, keeping the first occurrence. The three
    returned lists have identical length and ordering.

    Parameters
    ----------
    variables : list of TimeAwareSymbol
        Variables at any time index.

    Returns
    -------
    lags : list of TimeAwareSymbol
        Each variable at t-1.
    now : list of TimeAwareSymbol
        Each variable at t.
    leads : list of TimeAwareSymbol
        Each variable at t+1.
    """
    now = list(dict.fromkeys(x.set_t(0) for x in variables))
    lags = [x.step_backward() for x in now]
    leads = [x.step_forward() for x in now]
    return lags, now, leads
