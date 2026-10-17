import numpy as np
import pytest

from numpy.testing import assert_allclose

import gEconpy.solvers.sparse_root.direction as direction_module

from gEconpy import simulate
from gEconpy.exceptions import NonFiniteInitialGuessError
from gEconpy.model.perfect_foresight import extended_path, simulate_extended_path, solve_perfect_foresight
from gEconpy.solvers.sparse_root.direction import ChordDirection
from gEconpy.solvers.sparse_root.line_search import chord, newton_armijo
from tests._resources.cache_compiled_models import load_and_cache_example, load_and_cache_model


@pytest.fixture
def rbc_model():
    return load_and_cache_model("one_block_1.gcn")


@pytest.fixture
def linear_nk_model():
    return load_and_cache_model("3_eq_linear_nk.gcn")


def _steady_state_values(model, var_names, **parameter_updates):
    steady_state = model.steady_state(verbose=False, progressbar=False, **parameter_updates)
    return np.array([steady_state[f"{name}_ss"] for name in var_names])


@pytest.mark.parametrize(
    ("gcn", "parameter_updates"),
    [("one_block_1.gcn", {}), ("one_block_1.gcn", {"beta": 0.98}), ("open_rbc.gcn", {"gamma_rv": 2.0})],
    ids=["model_parameters", "updated_beta", "free_parameter_behind_a_deterministic_one"],
)
def test_zero_shocks_stay_at_the_steady_state(gcn, parameter_updates):
    model = load_and_cache_model(gcn)
    paths, diagnostics = simulate_extended_path(
        model, n_simulations=2, simulation_length=5, shock_std=0.0, steady_state_kwargs=parameter_updates
    )

    var_names = list(paths.coords["variable"].values)
    steady_state_values = _steady_state_values(model, var_names, **parameter_updates)
    assert diagnostics.success.shape == (2, 5)
    assert diagnostics.success.all()
    assert not diagnostics.nit.any()
    assert_allclose(paths.values, np.broadcast_to(steady_state_values, paths.shape))


def test_linear_model_reproduces_the_linear_simulator(linear_nk_model):
    simulation_kwargs = {"n_simulations": 2, "simulation_length": 20, "shock_std": 0.01, "random_seed": 3}

    paths, diagnostics = simulate_extended_path(linear_nk_model, **simulation_kwargs)
    linear_paths = simulate(linear_nk_model, verbose=False, **simulation_kwargs)

    assert diagnostics.success.all()
    assert_allclose(paths.values, linear_paths.sel(variable=paths.coords["variable"]).values, atol=1e-12)


def test_small_shocks_match_the_linear_simulator_to_first_order(rbc_model):
    shock_std = 1e-3
    simulation_kwargs = {"simulation_length": 30, "shock_std": shock_std, "random_seed": 1}

    # Capital's root of 0.98 needs a long horizon, or imposing the steady state at its end is a first-order error.
    paths, diagnostics = simulate_extended_path(
        rbc_model, horizon=400, sparse_root_kwargs={"tol": 1e-12}, **simulation_kwargs
    )
    linear_paths = simulate(rbc_model, verbose=False, **simulation_kwargs)

    var_names = list(paths.coords["variable"].values)
    log_deviations = np.log(paths.values / _steady_state_values(rbc_model, var_names))

    assert diagnostics.success.all()
    assert_allclose(log_deviations, linear_paths.sel(variable=var_names).values, atol=1e-2 * shock_std)


def test_zero_shocks_from_a_displaced_state_follow_the_perfect_foresight_path(rbc_model):
    """
    Without surprises each period's plan continues the last, so the path is one PF solution.

    The shifted warm start then leaves only the truncated horizon's last period to correct, in one Newton step.
    """
    steady_state = rbc_model.steady_state(verbose=False, progressbar=False)
    initial_conditions = {"K": 0.9 * steady_state["K_ss"]}

    paths, diagnostics = simulate_extended_path(
        rbc_model,
        simulation_length=10,
        shock_std=0.0,
        initial_conditions=initial_conditions,
        horizon=400,
        solver=newton_armijo(),
        sparse_root_kwargs={"tol": 1e-10},
    )
    pf_path, _ = solve_perfect_foresight(rbc_model, simulation_length=400, initial_conditions=initial_conditions)

    var_names = list(paths.coords["variable"].values)
    assert_allclose(paths.values[0], pf_path[var_names].to_numpy()[:10], rtol=1e-6)
    assert (diagnostics.nit.values[0, 1:] == 1).all()


def test_default_tolerance_leaves_the_stall_test_strict():
    """Newton takes many short steps on this model's first period, which a step tolerance of 1e-5 calls a stall."""
    model = load_and_cache_example("RBC_two_household_additive")

    # Only the solver's stall test matters here, so FAST_COMPILE skips the model's numba compilation.
    _, diagnostics = simulate_extended_path(
        model, simulation_length=1, shock_std=1e-4, random_seed=0, compile_kwargs={"mode": "FAST_COMPILE"}
    )

    assert diagnostics.success.all()


def test_the_default_solver_reuses_factorizations_across_periods(rbc_model, monkeypatch):
    simulation_kwargs = {"simulation_length": 10, "shock_std": 0.01, "random_seed": 1}
    expected, _ = simulate_extended_path(rbc_model, solver=newton_armijo(), **simulation_kwargs)

    calls = {"n": 0}
    real_splu = direction_module.splu

    def counting_splu(A, **kwargs):
        calls["n"] += 1
        return real_splu(A, **kwargs)

    monkeypatch.setattr(direction_module, "splu", counting_splu)
    paths, diagnostics = simulate_extended_path(rbc_model, **simulation_kwargs)

    assert diagnostics.success.all()
    assert calls["n"] < simulation_kwargs["simulation_length"]
    # Both runs solve only to f_tol = 1e-5, so they agree to about that.
    assert_allclose(paths.values, expected.values, rtol=1e-4)


def test_a_trajectory_does_not_depend_on_the_trajectories_before_it(rbc_model, monkeypatch):
    # Shocks large enough that the first trajectory refactorizes mid-run, so it ends with a factorization the second
    # trajectory would not start with.
    shocks = np.random.default_rng(0).normal(scale=[[[0.1]], [[0.01]]], size=(2, 10, len(rbc_model.shocks)))
    monkeypatch.setattr(extended_path, "_draw_shocks", lambda n_simulations, **_: shocks[-n_simulations:])

    both, _ = simulate_extended_path(rbc_model, n_simulations=2, simulation_length=10)
    second_alone, _ = simulate_extended_path(rbc_model, n_simulations=1, simulation_length=10)

    np.testing.assert_array_equal(both.values[1], second_alone.values[0])


def test_a_solver_holding_a_factorization_can_be_passed_in(rbc_model):
    solver = chord(direction=ChordDirection(keep_across_solves=True))
    steady_state = rbc_model.steady_state(verbose=False, progressbar=False)
    solve_perfect_foresight(
        rbc_model, simulation_length=50, solver=solver, initial_conditions={"K": 0.9 * steady_state["K_ss"]}
    )
    assert solver.direction._factorization.lu is not None

    _, diagnostics = simulate_extended_path(rbc_model, simulation_length=3, shock_std=0.01, solver=solver)

    assert diagnostics.success.all()


def test_a_tight_tolerance_needs_no_homotopy_with_the_default_solver():
    """A carried factorization converges linearly, so its last steps fall below x_tol before the residual meets tol."""
    model = load_and_cache_model("full_nk.gcn")

    _, diagnostics = simulate_extended_path(
        model,
        simulation_length=3,
        shock_std=0.01,
        random_seed=0,
        horizon=50,
        sparse_root_kwargs={"tol": 1e-10},
        compile_kwargs={"mode": "FAST_COMPILE"},
    )

    assert diagnostics.success.all()
    assert not diagnostics.homotopy_solves.any()


def test_a_failed_period_ends_only_its_own_trajectory(rbc_model):
    # One Newton iteration is too few for some of these shocks, so each trajectory fails at its own period.
    with pytest.warns(UserWarning, match="stopped at period") as record:
        paths, diagnostics = simulate_extended_path(
            rbc_model,
            n_simulations=2,
            simulation_length=8,
            shock_std=0.01,
            random_seed=1,
            sparse_root_kwargs={"maxiter": 1},
            max_homotopy_solves=0,
        )

    failed_periods = diagnostics.success.argmin(dim="time").values
    assert list(failed_periods) == [1, 3]
    assert [str(warning.message).split(",")[0] for warning in record] == [
        "Simulation 0 stopped at period 1",
        "Simulation 1 stopped at period 3",
    ]
    for simulation, failed_period in enumerate(failed_periods):
        assert np.isfinite(paths.values[simulation, :failed_period]).all()
        assert np.isnan(paths.values[simulation, failed_period:]).all()
        assert not diagnostics.success.values[simulation, failed_period:].any()
        assert not diagnostics.nit.values[simulation, failed_period + 1 :].any()


def test_a_non_finite_warm_start_is_retried_from_the_steady_state(rbc_model, monkeypatch):
    simulation_kwargs = {"simulation_length": 4, "shock_std": 0.01, "random_seed": 0, "horizon": 100}
    expected, _ = simulate_extended_path(rbc_model, **simulation_kwargs)

    calls = []
    real_solve_stacked = extended_path._solve_stacked

    def warm_start_at_period_two_is_not_finite(*args, **kwargs):
        calls.append(kwargs["x0"])
        if len(calls) == 3:
            raise NonFiniteInitialGuessError(equation=0, period=0)
        return real_solve_stacked(*args, **kwargs)

    monkeypatch.setattr(extended_path, "_solve_stacked", warm_start_at_period_two_is_not_finite)
    paths, diagnostics = simulate_extended_path(rbc_model, **simulation_kwargs)

    assert diagnostics.success.all()
    assert_allclose(calls[3], calls[0])
    assert_allclose(paths.values, expected.values, rtol=1e-8)


@pytest.mark.parametrize(
    "max_homotopy_solves, most_solves_spent", [(5, 5), (1000, 999)], ids=["budget_runs_out", "step_shrinks_away"]
)
def test_a_state_where_the_model_is_not_finite_ends_the_trajectory(rbc_model, max_homotopy_solves, most_solves_spent):
    with pytest.warns(UserWarning, match="Simulation 0 stopped at period 0.*Homotopy reached weight"):
        paths, diagnostics = simulate_extended_path(
            rbc_model,
            simulation_length=3,
            shock_std=0.0,
            initial_conditions={"K": -1.0},
            horizon=50,
            max_homotopy_solves=max_homotopy_solves,
        )

    assert np.isnan(paths.values).all()
    assert not diagnostics.success.any()
    assert 0 < diagnostics.homotopy_solves.values[0, 0] <= most_solves_spent


def test_periods_that_fail_under_an_iteration_cap_succeed_by_homotopy(rbc_model):
    simulation_kwargs = {"n_simulations": 2, "simulation_length": 8, "shock_std": 0.01, "random_seed": 1}
    expected, _ = simulate_extended_path(rbc_model, **simulation_kwargs)

    paths, diagnostics = simulate_extended_path(rbc_model, sparse_root_kwargs={"maxiter": 1}, **simulation_kwargs)

    assert diagnostics.success.all()
    assert diagnostics.homotopy_solves.any()
    assert (diagnostics.nsolve.values >= diagnostics.homotopy_solves.values).all()
    # Every period solves only to f_tol = 1e-5, so the two runs agree to about that.
    assert_allclose(paths.values, expected.values, rtol=1e-4)


def test_a_deterministic_parameter_update_raises():
    model = load_and_cache_model("open_rbc.gcn")

    with pytest.raises(ValueError, match="Cannot set a path for deterministic parameters: gamma"):
        simulate_extended_path(model, simulation_length=2, shock_std=0.0, steady_state_kwargs={"gamma": 1.0})


def test_unknown_initial_condition_raises(rbc_model):
    with pytest.raises(ValueError, match="unknown variables: Q"):
        simulate_extended_path(rbc_model, shock_std=0.0, initial_conditions={"Q": 1.0})
