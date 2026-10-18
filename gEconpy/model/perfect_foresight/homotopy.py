from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import numpy as np

from scipy.optimize import OptimizeResult

from gEconpy.exceptions import NonFiniteInitialGuessError
from gEconpy.model.perfect_foresight._parallel import _WORK_COUNT_FIELDS

_HOMOTOPY_INITIAL_STEP = 0.1
_HOMOTOPY_MIN_STEP = 1e-4


class _HomotopyOutcome(NamedTuple):
    path: np.ndarray | None
    full_weight_result: OptimizeResult | None
    work_counts: dict[str, int]
    homotopy_solves: int
    failure: str | None
    non_finite_guess: NonFiniteInitialGuessError | None


def _solve_by_homotopy(
    solve: Callable[..., tuple[np.ndarray, OptimizeResult]],
    warm_guess: np.ndarray,
    base_guess: Callable[[], np.ndarray],
    max_homotopy_solves: int,
) -> _HomotopyOutcome:
    """
    Solve a problem at full weight from ``warm_guess``, and by homotopy from ``base_guess`` if that fails.

    ``solve(weight=..., x0=...)`` solves the problem with its departure from the steady state scaled by ``weight``,
    so weight 0 is close to the steady state and weight 1 is the problem itself. The weight rises from 0 to 1, each
    solve starting from the last solution. The step doubles after a success and halves after a failure.

    Parameters
    ----------
    solve : callable
        Takes ``weight`` and ``x0`` and returns the solution and the solver's result.
    warm_guess : ndarray
        Starting point of the first, full-weight attempt.
    base_guess : callable
        Returns the starting point of the homotopy at weight 0, which is also the fallback when a guess is not
        finite. It is called only when one of the two is needed, and at most once.
    max_homotopy_solves : int
        Number of solves the homotopy may spend after the full-weight attempt fails. Zero turns it off.

    Returns
    -------
    outcome : _HomotopyOutcome
        The solution, or None if the homotopy did not reach weight 1. ``full_weight_result`` is the result of the
        last solve at weight 1, and None when every attempt at weight 1 found the model not finite at its guess.
        ``non_finite_guess`` holds the last such error, and the work counts are summed over every solve.
    """
    base_guess = cache(base_guess)
    work_counts: dict[str, int] = dict.fromkeys(_WORK_COUNT_FIELDS, 0)
    full_weight_result = None
    non_finite_guess = None

    def attempt(weight: float, guess: np.ndarray) -> tuple[np.ndarray | None, str | None]:
        nonlocal full_weight_result, non_finite_guess
        try:
            solution, result = _solve_from(solve, weight=weight, guess=guess, base_guess=base_guess)
        except NonFiniteInitialGuessError as error:
            non_finite_guess = error
            return None, str(error)

        for field in _WORK_COUNT_FIELDS:
            work_counts[field] += result[field]
        if weight == 1.0:
            full_weight_result = result
        if not result.success:
            return None, f"The solver reported: {result.message}"
        return solution, None

    def outcome(path: np.ndarray | None, homotopy_solves: int, failure: str | None) -> _HomotopyOutcome:
        return _HomotopyOutcome(
            path=path,
            full_weight_result=full_weight_result,
            work_counts=work_counts,
            homotopy_solves=homotopy_solves,
            failure=failure,
            non_finite_guess=non_finite_guess,
        )

    path, failure = attempt(weight=1.0, guess=warm_guess)
    if path is not None:
        return outcome(path, homotopy_solves=0, failure=None)

    accepted_weight = 0.0
    step = _HOMOTOPY_INITIAL_STEP
    guess = base_guess()
    homotopy_solves = 0
    while homotopy_solves < max_homotopy_solves and step >= _HOMOTOPY_MIN_STEP:
        weight = min(accepted_weight + step, 1.0)
        homotopy_solves += 1
        path, _ = attempt(weight=weight, guess=guess)

        if path is None:
            step /= 2
        elif weight == 1.0:
            return outcome(path, homotopy_solves=homotopy_solves, failure=None)
        else:
            accepted_weight, guess = weight, path
            step *= 2

    failure = f"{failure.rstrip('.')}. Homotopy reached weight {accepted_weight:.3g} in {homotopy_solves} solves."
    return outcome(None, homotopy_solves=homotopy_solves, failure=failure)


def _solve_from(
    solve: Callable[..., tuple[np.ndarray, OptimizeResult]],
    weight: float,
    guess: np.ndarray,
    base_guess: Callable[[], np.ndarray],
) -> tuple[np.ndarray, OptimizeResult]:
    try:
        return solve(weight=weight, x0=guess)
    except NonFiniteInitialGuessError:
        if guess is base_guess():
            raise
        return solve(weight=weight, x0=base_guess())
