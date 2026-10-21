import numpy as np
import pytensor
import pytest

from numpy.testing import assert_allclose
from pytensor.gradient import verify_grad

from gEconpy.solvers.second_order import solve_second_order
from tests._resources.cache_compiled_models import load_and_cache_model
from tests._resources.load_dynare import extract_second_order_matrices, read_dynare_output


def solve_second_order_numerically(model, log_linearize):
    kwargs = {"log_linearize": log_linearize, "verbose": False}
    jacobians, ss_inputs, param_inputs, _, _ = model.symbolic_linearization(**kwargs)
    derivatives, composition_params = model.second_order_derivative_terms(**kwargs)
    terms = solve_second_order(
        jacobians=jacobians,
        composition_hessian=derivatives.hessian,
        ghx=derivatives.ghx,
        ghu=derivatives.ghu,
        dr_order=model.dr_order,
    )

    inputs = list(dict.fromkeys([*ss_inputs, *derivatives.ss_inputs, *param_inputs, *composition_params]))
    f = pytensor.function([derivatives.ghx, derivatives.ghu, *inputs], list(terms), on_unused_input="ignore")

    solution = model.solve_model(return_matrices=False, **kwargs)
    values = {**model.steady_state(verbose=False, progressbar=False).to_string(), **model.parameters()}
    return f(solution.ghx, solution.ghu, *[values[node.name] for node in inputs])


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
def test_second_order_solution_matches_dynare(gcn_name, dynare_name, log_linearize):
    model = load_and_cache_model(gcn_name + ".gcn")
    ghxx, ghxu, ghuu, ghs2_operator = solve_second_order_numerically(model, log_linearize)

    dynare = extract_second_order_matrices(*read_dynare_output(dynare_name))

    # Dynare's var(log) declares LOG_X beside a static X, so the log model's rows are found by name.
    prefix = "LOG_" if log_linearize else ""
    names = [prefix + model.variables[i].base_name for i in model.dr_order.var_order]
    rows = [dynare.names.index(name) for name in names]
    states = [dynare.state_names.index(names[i]) for i in model.dr_order.lag_positions]

    assert_allclose(ghxx, dynare.ghxx[rows][:, states][:, :, states], atol=1e-8)
    assert_allclose(ghxu, dynare.ghxu[rows][:, states], atol=1e-8)
    assert_allclose(ghuu, dynare.ghuu[rows], atol=1e-8)
    assert_allclose(ghs2_operator @ dynare.shock_cov.ravel(), dynare.ghs2[rows], atol=1e-10)


def test_second_order_gradients_match_finite_differences():
    dr_order = load_and_cache_model("one_block_1_ss_2shock.gcn").dr_order
    n_variables, n_states, n_shocks = len(dr_order.var_order), dr_order.n_state_var, 2
    n_w = n_states + 2 * n_shocks
    rng = np.random.default_rng(0)

    lag_jacobian = rng.normal(scale=0.3, size=(n_variables, n_variables))
    shock_jacobian = rng.normal(size=(n_variables, n_shocks))
    now_jacobian = np.eye(n_variables) + rng.normal(scale=0.1, size=(n_variables, n_variables))
    lead_jacobian = rng.normal(scale=0.1, size=(n_variables, n_variables))
    hessian = rng.normal(size=(n_variables, n_w, n_w))
    ghx = rng.normal(scale=0.3, size=(n_variables, n_states))
    ghu = rng.normal(scale=0.3, size=(n_variables, n_shocks))

    def total(now_jacobian, lead_jacobian, hessian, ghx, ghu):
        jacobians = [lag_jacobian, now_jacobian, lead_jacobian, shock_jacobian]
        symmetric_hessian = hessian + hessian.transpose(0, 2, 1)
        terms = solve_second_order(
            jacobians=jacobians, composition_hessian=symmetric_hessian, ghx=ghx, ghu=ghu, dr_order=dr_order
        )
        return sum(term.sum() for term in terms)

    verify_grad(total, [now_jacobian, lead_jacobian, hessian, ghx, ghu], rng=rng)
