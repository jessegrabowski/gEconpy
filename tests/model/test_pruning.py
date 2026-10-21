import numpy as np
import pytest

from numpy.testing import assert_allclose
from pytensor.gradient import verify_grad
from scipy import linalg

from gEconpy.model.pruning import (
    compile_pruned_second_order_path,
    compile_pruned_second_order_system,
    pruned_second_order_path,
    pruned_second_order_system,
)
from gEconpy.model.simulate import _simulate_linear_system, _simulate_pruned_system
from tests._resources.cache_compiled_models import load_and_cache_model
from tests._resources.load_dynare import read_pruned_path


def test_pruned_path_without_second_order_terms_is_the_first_order_path():
    solution = load_and_cache_model("one_block_1_ss_2shock.gcn").solve_model(order=2, verbose=False)
    shocks = np.random.default_rng(0).normal(scale=0.01, size=(3, 30, 2))

    n_variables, n_states = solution.ghx.shape
    n_shocks = solution.ghu.shape[1]
    deviations = compile_pruned_second_order_path()(
        ghx=solution.ghx,
        ghu=solution.ghu,
        ghxx=np.zeros((n_variables, n_states, n_states)),
        ghxu=np.zeros((n_variables, n_states, n_shocks)),
        ghuu=np.zeros((n_variables, n_shocks, n_shocks)),
        ghs2=np.zeros(n_variables),
        shocks=shocks,
        state_positions=solution.dr_order.lag_positions,
    )

    expected = _simulate_linear_system(solution.T, solution.R, shocks)
    assert_allclose(deviations[..., solution.dr_order.inv_var_order], expected, atol=1e-14)


@pytest.mark.parametrize(
    ("gcn_name", "dynare_name", "log_linearize"),
    [
        ("one_block_1_ss", "one_block_1_ss", False),
        ("one_block_1_ss_2shock", "one_block_1_ss_2shock", False),
        ("rbc_2_block_ss", "rbc_2_block_ss", False),
        ("basic_rbc", "basic_rbc", False),
        ("basic_rbc", "basic_rbc_logvars", True),
        pytest.param("full_nk", "full_nk", False, marks=pytest.mark.include_nk),
    ],
    ids=["one_block_1_ss", "one_block_1_ss_2shock", "rbc_2_block_ss", "basic_rbc", "basic_rbc_logvars", "full_nk"],
)
def test_pruned_path_matches_dynare(gcn_name, dynare_name, log_linearize):
    model = load_and_cache_model(gcn_name + ".gcn")
    solution = model.solve_model(order=2, log_linearize=log_linearize, verbose=False)
    shocks, shock_cov, dynare_deviations = read_pruned_path(dynare_name)

    # Dynare's var(log) declares LOG_X beside a static X, so the log model's columns are found by name.
    prefix = "LOG_" if log_linearize else ""
    columns = [prefix + name for name in solution.variable_names]
    deviations = _simulate_pruned_system(solution, shocks, ghs2=solution.ghs2(shock_cov))

    assert_allclose(deviations, dynare_deviations[columns].to_numpy(), atol=1e-9)


def test_pruned_path_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    n_variables, n_states, n_shocks, n_periods = 3, 2, 2, 4
    shapes = [
        (n_variables, n_states),
        (n_variables, n_shocks),
        (n_variables, n_states, n_states),
        (n_variables, n_states, n_shocks),
        (n_variables, n_shocks, n_shocks),
        (n_variables,),
        (n_periods, n_shocks),
    ]

    def path(*args):
        return pruned_second_order_path(*args, state_positions=np.array([0, 2]))

    verify_grad(path, [rng.normal(scale=0.5, size=shape) for shape in shapes], rng=rng)


def _solution_inputs(solution):
    return {
        "ghx": solution.ghx,
        "ghu": solution.ghu,
        "ghxx": solution.ghxx,
        "ghxu": solution.ghxu,
        "ghuu": solution.ghuu,
        "ghs2_operator": solution.ghs2_operator,
        "state_positions": solution.dr_order.lag_positions,
    }


def test_pruned_system_reproduces_the_pruned_path():
    solution = load_and_cache_model("one_block_1_ss_2shock.gcn").solve_model(order=2, verbose=False)
    shock_cov = np.array([[1e-4, 2e-5], [2e-5, 4e-4]])
    shocks = np.random.default_rng(0).multivariate_normal(np.zeros(2), shock_cov, size=30)
    n_variables = len(solution.variable_names)
    states = solution.dr_order.lag_positions
    shock_pairs = np.tril_indices(2)

    intercept, transition, selection, _ = compile_pruned_second_order_system()(
        **_solution_inputs(solution), shock_cov=shock_cov
    )

    state = np.zeros(transition.shape[0])
    deviations = []
    for shock in shocks:
        lagged_states = state[:n_variables][states]
        shock_square = (np.outer(shock, shock) - shock_cov)[shock_pairs]
        innovation = np.concatenate([shock, shock_square, np.kron(lagged_states, shock)])
        state = intercept + transition @ state + selection @ innovation
        deviations.append(state[:n_variables] + state[n_variables : 2 * n_variables])

    expected = _simulate_pruned_system(solution, shocks, ghs2=solution.ghs2(shock_cov))
    assert_allclose(np.array(deviations)[:, solution.dr_order.inv_var_order], expected, atol=1e-13)


def test_pruned_innovations_have_the_stated_covariance():
    """
    The innovations combine Gaussian shocks with first-order states drawn from their stationary distribution.

    With full-rank first-order states every innovation is a distinct combination, so the covariance is also positive
    definite.
    """
    solution = load_and_cache_model("one_block_1_ss_2shock.gcn").solve_model(order=2, verbose=False)
    shock_cov = np.array([[1e-4, 2e-5], [2e-5, 4e-4]])
    *_, innovation_cov = compile_pruned_second_order_system()(**_solution_inputs(solution), shock_cov=shock_cov)

    state_transition = solution.ghx[solution.dr_order.lag_positions]
    state_impact = solution.ghu[solution.dr_order.lag_positions]
    state_cov = linalg.solve_discrete_lyapunov(state_transition, state_impact @ shock_cov @ state_impact.T)

    rng = np.random.default_rng(0)
    n_draws = 400_000
    shocks = rng.multivariate_normal(np.zeros(2), shock_cov, size=n_draws)
    lagged_states = rng.multivariate_normal(np.zeros(len(state_cov)), state_cov, size=n_draws)
    innovations = np.hstack(
        [
            shocks,
            (np.einsum("ti,tj->tij", shocks, shocks) - shock_cov)[:, *np.tril_indices(2)],
            np.einsum("ti,tj->tij", lagged_states, shocks).reshape(n_draws, -1),
        ]
    )
    sample_cov = innovations.T @ innovations / n_draws

    scale = np.sqrt(np.outer(np.diag(innovation_cov), np.diag(innovation_cov)))
    assert_allclose(sample_cov / scale, innovation_cov / scale, atol=0.02)
    assert np.linalg.eigvalsh(innovation_cov).min() > 0


@pytest.mark.parametrize(
    "total",
    [
        lambda *inputs: sum(output.sum() for output in pruned_second_order_system(*inputs)),
    ],
    ids=["system"],
)
def test_pruned_graph_gradients_match_finite_differences(total):
    rng = np.random.default_rng(0)
    n_variables, n_states, n_shocks = 3, 2, 2
    shapes = [
        (n_variables, n_states),
        (n_variables, n_shocks),
        (n_variables, n_states, n_states),
        (n_variables, n_states, n_shocks),
        (n_variables, n_shocks, n_shocks),
        (n_variables, n_shocks**2),
    ]

    def graph(ghx, ghu, ghxx, ghxu, ghuu, ghs2_operator, shock_factor):
        shock_cov = shock_factor @ shock_factor.T
        return total(ghx, ghu, ghxx, ghxu, ghuu, ghs2_operator, shock_cov, np.array([0, 2]))

    point = [rng.normal(scale=0.3, size=shape) for shape in shapes] + [np.eye(n_shocks) + 0.1]
    verify_grad(graph, point, rng=rng)
