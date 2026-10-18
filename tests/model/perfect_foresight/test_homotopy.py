import numpy as np
import pytest

from scipy.optimize import OptimizeResult

from gEconpy.exceptions import NonFiniteInitialGuessError
from gEconpy.model.perfect_foresight.homotopy import _solve_by_homotopy


def _result(success: bool, weight: float) -> OptimizeResult:
    return OptimizeResult(
        x=np.array([weight]),
        success=success,
        message="no step",
        nit=1,
        nfev=1,
        njev=1,
        nsolve=1,
    )


def _steps_of_at_most(largest_step: float, weights: list[float]):
    """Return a solve that converges only when the weight is within ``largest_step`` of the guess's weight."""

    def solve(weight: float, x0: np.ndarray) -> tuple[np.ndarray, OptimizeResult]:
        weights.append(weight)
        result = _result(success=weight - x0[0] <= largest_step + 1e-12, weight=weight)
        return result.x, result

    return solve


def test_the_step_doubles_after_a_success_and_halves_after_a_failure():
    weights = []

    outcome = _solve_by_homotopy(
        solve=_steps_of_at_most(0.3, weights),
        warm_guess=np.array([0.0]),
        base_guess=lambda: np.array([0.0]),
        max_homotopy_solves=100,
    )

    assert weights == pytest.approx([1.0, 0.1, 0.3, 0.7, 0.5, 0.9, 0.7, 1.0])
    assert outcome.path == pytest.approx([1.0])
    assert outcome.homotopy_solves == 7
    assert outcome.work_counts == {"nit": 8, "nfev": 8, "njev": 8, "nsolve": 8}
    assert outcome.failure is None


@pytest.mark.parametrize(
    "max_homotopy_solves, expected_solves", [(3, 3), (1000, 10)], ids=["budget_runs_out", "step_shrinks_away"]
)
def test_an_unreachable_weight_ends_the_homotopy(max_homotopy_solves, expected_solves):
    outcome = _solve_by_homotopy(
        solve=_steps_of_at_most(0.0, []),
        warm_guess=np.array([0.0]),
        base_guess=lambda: np.array([0.0]),
        max_homotopy_solves=max_homotopy_solves,
    )

    assert outcome.path is None
    assert outcome.homotopy_solves == expected_solves
    assert outcome.failure == (f"The solver reported: no step. Homotopy reached weight 0 in {expected_solves} solves.")
    assert not outcome.full_weight_result.success


def test_the_base_guess_is_not_built_when_the_warm_guess_solves():
    def base_guess():
        raise AssertionError("base guess built")

    outcome = _solve_by_homotopy(
        solve=_steps_of_at_most(1.0, []),
        warm_guess=np.array([0.0]),
        base_guess=base_guess,
        max_homotopy_solves=100,
    )

    assert outcome.path == pytest.approx([1.0])


def test_a_non_finite_guess_is_retried_from_the_base_guess():
    base = np.array([0.0])

    def solve(weight: float, x0: np.ndarray) -> tuple[np.ndarray, OptimizeResult]:
        if x0 is not base:
            raise NonFiniteInitialGuessError(equation=0, period=0)
        result = _result(success=True, weight=weight)
        return result.x, result

    outcome = _solve_by_homotopy(
        solve=solve, warm_guess=np.array([np.inf]), base_guess=lambda: base, max_homotopy_solves=100
    )

    assert outcome.path == pytest.approx([1.0])
    assert outcome.homotopy_solves == 0


def test_a_guess_that_is_never_finite_is_reported_without_a_full_weight_result():
    def solve(**_) -> tuple[np.ndarray, OptimizeResult]:
        raise NonFiniteInitialGuessError(equation=2, period=5)

    outcome = _solve_by_homotopy(
        solve=solve, warm_guess=np.array([0.0]), base_guess=lambda: np.array([0.0]), max_homotopy_solves=0
    )

    assert outcome.full_weight_result is None
    assert str(outcome.non_finite_guess) == str(NonFiniteInitialGuessError(equation=2, period=5))
    assert outcome.work_counts == {"nit": 0, "nfev": 0, "njev": 0, "nsolve": 0}
