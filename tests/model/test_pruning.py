import numpy as np
import pytest

from numpy.testing import assert_allclose
from pytensor.gradient import verify_grad

from gEconpy.model.pruning import compile_pruned_second_order_path, pruned_second_order_path
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
