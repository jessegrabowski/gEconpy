import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_array_equal

from gEconpy.classes.perturbation_solution import FirstOrderPerturbationSolution
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.perturbation import DROrder


@pytest.fixture
def dr_order():
    static, pred, mixed, fwd = (TimeAwareSymbol(name, 0) for name in ["static", "pred", "mixed", "fwd"])
    equations = [
        fwd.set_t(1) - mixed,
        static - pred,
        pred - 0.5 * pred.set_t(-1),
        mixed.set_t(1) - mixed.set_t(-1) + static,
    ]
    return DROrder.from_model([fwd, mixed, pred, static], equations)


def _solution(T, R, dr_order, log_linearized=None, steady_state=None):
    n_variables = T.shape[0]
    return FirstOrderPerturbationSolution.from_policy_matrices(
        T=T,
        R=R,
        steady_state=np.ones(n_variables) if steady_state is None else steady_state,
        log_linearized=np.zeros(n_variables, dtype=bool) if log_linearized is None else log_linearized,
        variable_names=["fwd", "mixed", "pred", "static"],
        shock_names=["epsilon"],
        dr_order=dr_order,
    )


def test_policy_matrices_round_trip_through_the_state_columns(dr_order):
    rng = np.random.default_rng(0)
    state_columns = dr_order.var_order[dr_order.lag_positions]
    T = np.zeros((4, 4))
    T[:, state_columns] = rng.normal(size=(4, len(state_columns)))
    R = rng.normal(size=(4, 1))

    solution = _solution(T, R, dr_order)
    T_out, R_out = solution

    assert solution.ghx.shape == (4, dr_order.n_state_var)
    assert_array_equal(solution.ghx, T[np.ix_(dr_order.var_order, state_columns)])
    assert_array_equal(T_out, T)
    assert_array_equal(R_out, R)


def test_to_levels_exponentiates_only_log_linearized_variables(dr_order):
    steady_state = np.array([2.0, 3.0, -1.0, 0.0])
    log_linearized = np.array([True, True, False, False])
    solution = _solution(np.zeros((4, 4)), np.zeros((4, 1)), dr_order, log_linearized, steady_state)
    deviations = np.array([[0.1, -0.2, 0.5, 0.3], [0.0, 0.0, 0.0, 0.0]])

    levels = solution.to_levels(deviations)

    expected = np.array([[2.0 * np.exp(0.1), 3.0 * np.exp(-0.2), -0.5, 0.3], steady_state])
    assert_allclose(levels, expected)
