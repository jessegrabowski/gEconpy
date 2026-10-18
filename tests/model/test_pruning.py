import numpy as np

from numpy.testing import assert_allclose
from pytensor.gradient import verify_grad

from gEconpy.model.pruning import compile_pruned_second_order_path, pruned_second_order_path
from gEconpy.model.simulate import _simulate_linear_system
from tests._resources.cache_compiled_models import load_and_cache_model


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
