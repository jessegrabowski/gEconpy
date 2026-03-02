import re

from functools import reduce

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol


def natural_sort_key(symbol: TimeAwareSymbol) -> list:
    """Sort key that orders numeric suffixes numerically.

    Sorts ``x1, x2, x10`` correctly rather than lexicographically (``x1, x10, x2``).

    Parameters
    ----------
    symbol : TimeAwareSymbol
        Symbol whose ``base_name`` is used for sorting.

    Returns
    -------
    key : list
        Mixed list of strings and ints suitable for use as a sort key.
    """
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", symbol.base_name)]


def collect_time_aware_atoms(equations: list[sp.Expr]) -> set[TimeAwareSymbol]:
    """Collect all :class:`TimeAwareSymbol` atoms from a list of sympy expressions.

    Parameters
    ----------
    equations : list of sp.Expr
        Sympy expressions to scan.

    Returns
    -------
    atoms : set of TimeAwareSymbol
    """
    return reduce(lambda a, b: a | b, (eq.atoms(TimeAwareSymbol) for eq in equations), set())


def classify_variables_by_timing(
    equations: list[sp.Expr],
    shock_names: list[str],
) -> tuple[list[TimeAwareSymbol], list[TimeAwareSymbol], list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    """Extract variables and shocks from equations and group by time index.

    Returns only the symbols that actually appear at each time index. Results are sorted with
    :func:`natural_sort_key`.

    Parameters
    ----------
    equations : list of sp.Expr
        Model equations as sympy expressions.
    shock_names : list of str
        Base names of exogenous shocks.

    Returns
    -------
    vars_tm1 : list of TimeAwareSymbol
        Variables at t-1.
    vars_t : list of TimeAwareSymbol
        Variables at t.
    vars_tp1 : list of TimeAwareSymbol
        Variables at t+1.
    shocks_t : list of TimeAwareSymbol
        Shocks (any time index).
    """
    all_atoms = collect_time_aware_atoms(equations)
    shock_name_set = set(shock_names)

    VALID_TIME_INDICES = {-1, 0, 1}
    invalid = {x for x in all_atoms if x.time_index not in VALID_TIME_INDICES}
    if invalid:
        bad = ", ".join(f"{x.name} (t={x.time_index})" for x in sorted(invalid, key=natural_sort_key))
        raise ValueError(
            f"Equations contain variables at unexpected time indices. Expected only t-1, t, or t+1, "
            f"found: {bad}. Equations should be normalized before classification."
        )

    atoms_by_time: dict[tuple[str, int], TimeAwareSymbol] = {
        (x.base_name, x.time_index): x for x in all_atoms if x.base_name not in shock_name_set
    }

    def vars_at_time(t: int) -> list[TimeAwareSymbol]:
        return sorted(
            [sym for (_, ti), sym in atoms_by_time.items() if ti == t],
            key=natural_sort_key,
        )

    shocks = sorted([x for x in all_atoms if x.base_name in shock_name_set], key=natural_sort_key)
    return vars_at_time(-1), vars_at_time(0), vars_at_time(1), shocks


def make_all_variable_time_combinations(
    variables: list[TimeAwareSymbol],
) -> tuple[list[TimeAwareSymbol], list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    """Produce the t-1, t, and t+1 variants of every variable.

    Each input variable is first normalized to t=0. Duplicates (by base name) are removed,
    preserving the order of first occurrence. The three returned lists have identical length and
    ordering.

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
