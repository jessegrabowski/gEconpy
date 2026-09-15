import re

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import sympy as sp

from numpy.testing import assert_allclose
from pymc.distributions.transforms import Interval, log, logodds
from scipy import optimize

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.model import (
    Model,
    infer_variable_bounds,
    infer_variable_transform,
    transform_steady_state_system,
)
from gEconpy.model.parameters import compile_param_dict_func
from gEconpy.model.steady_state import (
    compile_known_ss,
    print_steady_state,
    propagate_steady_state_through_identities,
    pt_error_from_resid,
    system_to_steady_state,
)
from tests._resources.cache_compiled_models import load_and_cache_model


class TestPropagateSteadyStateThroughIdentities:
    def test_propagates_through_lag_chain(self):
        """Chained identities x__lag1 = x[-1], x__lag2 = x__lag1[-1] all resolve to x_ss."""
        x = TimeAwareSymbol("x", 0)
        x_lag1 = TimeAwareSymbol("x__lag1", 0)
        x_lag2 = TimeAwareSymbol("x__lag2", 0)

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary({x.to_ss(): sp.Float(0.0)}),
            steady_state_equations=[x_lag1.to_ss() - x.to_ss(), x_lag2.to_ss() - x_lag1.to_ss()],
            variables=[x, x_lag1, x_lag2],
        )

        assert len(result) == 3
        assert all(float(result[v.to_ss()]) == 0.0 for v in [x, x_lag1, x_lag2])

    def test_propagates_affine_relationships(self):
        """Affine equation y = 2*x + 3 is solved when x is known."""
        x = TimeAwareSymbol("x", 0)
        y = TimeAwareSymbol("y", 0)

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary({x.to_ss(): sp.Float(1.0)}),
            steady_state_equations=[y.to_ss() - 2 * x.to_ss() - 3],
            variables=[x, y],
        )

        assert float(result[y.to_ss()]) == 5.0

    def test_propagates_log_exp_bijections(self):
        """Bijection log(y) = x is inverted to y = exp(x)."""
        x = TimeAwareSymbol("x", 0)
        y = TimeAwareSymbol("y", 0)

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary({x.to_ss(): sp.Float(0.0)}),
            steady_state_equations=[sp.log(y.to_ss()) - x.to_ss()],
            variables=[x, y],
        )

        assert float(result[y.to_ss()]) == 1.0

    def test_rejects_multi_solution_equations(self):
        """Equation x^2 = 4 has two solutions, so neither is chosen."""
        x = TimeAwareSymbol("x", 0)
        y = TimeAwareSymbol("y", 0)

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary({y.to_ss(): sp.Float(4.0)}),
            steady_state_equations=[x.to_ss() ** 2 - y.to_ss()],
            variables=[x, y],
        )

        assert x.to_ss() not in result

    def test_rejects_underdetermined_equations(self):
        """Equation x + y = z with only z known cannot determine x or y."""
        x = TimeAwareSymbol("x", 0)
        y = TimeAwareSymbol("y", 0)
        z = TimeAwareSymbol("z", 0)

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary({z.to_ss(): sp.Float(5.0)}),
            steady_state_equations=[x.to_ss() + y.to_ss() - z.to_ss()],
            variables=[x, y, z],
        )

        assert x.to_ss() not in result
        assert y.to_ss() not in result

    def test_rejects_complex_ces_production_function(self):
        """CES production function inversion is too complex to attempt."""
        Y = TimeAwareSymbol("Y", 0)
        A = TimeAwareSymbol("A", 0)
        x1 = TimeAwareSymbol("x1", 0)
        x2 = TimeAwareSymbol("x2", 0)
        alpha, psi = sp.Symbol("alpha"), sp.Symbol("psi")

        ces_aggregator = (
            alpha ** (1 / psi) * x1.to_ss() ** ((psi - 1) / psi)
            + (1 - alpha) ** (1 / psi) * x2.to_ss() ** ((psi - 1) / psi)
        ) ** (psi / (psi - 1))

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary(
                {
                    Y.to_ss(): sp.Float(1.0),
                    A.to_ss(): sp.Float(1.0),
                    x1.to_ss(): sp.Float(0.5),
                }
            ),
            steady_state_equations=[Y.to_ss() - A.to_ss() * ces_aggregator],
            variables=[Y, A, x1, x2],
        )

        assert x2.to_ss() not in result

    def test_empty_input_with_underdetermined_system(self):
        """Empty input with multiple unknowns per equation returns empty."""
        x = TimeAwareSymbol("x", 0)
        y = TimeAwareSymbol("y", 0)

        result = propagate_steady_state_through_identities(SymbolDictionary(), [x.to_ss() + y.to_ss()], [x, y])
        assert len(result) == 0

    def test_solves_ar1_log_process(self):
        """AR(1) in logs: log(A) = rho*log(A[-1]) + epsilon solves to A_ss = 1 when epsilon_ss = 0."""
        A = TimeAwareSymbol("A", 0)
        rho = sp.Symbol("rho")

        steady_state_eq = sp.log(A.to_ss()) - rho * sp.log(A.to_ss())

        result = propagate_steady_state_through_identities(
            ss_solution_dict=SymbolDictionary(),
            steady_state_equations=[steady_state_eq],
            variables=[A],
        )

        assert A.to_ss() in result
        assert float(result[A.to_ss()]) == 1.0


def _collapse_whitespace(text: str) -> str:
    return re.sub(" +", " ", re.sub("[\t\n]", " ", text))


def assert_root_and_minimize_agree(model: Model, **steady_state_kwargs):
    ss_root = model.steady_state(how="root", verbose=False, progressbar=False, **steady_state_kwargs)
    ss_minimize = model.steady_state(how="minimize", verbose=False, progressbar=False, **steady_state_kwargs)

    assert ss_root.success
    assert ss_minimize.success

    for k in ss_root:
        assert_allclose(ss_root[k], ss_minimize[k], err_msg=k)


def test_solve_ss_with_partial_user_solution():
    model_1 = load_and_cache_model("one_block_1.gcn")
    res = model_1.steady_state(verbose=False, progressbar=False)
    assert res.success


def test_system_to_steady_state_collapses_time_indices_and_zeros_shocks():
    x, eps = TimeAwareSymbol("x", 0), TimeAwareSymbol("eps", 0)
    rho = sp.Symbol("rho")

    (steady_state_eq,) = system_to_steady_state([x - rho * x.set_t(-1) - eps + x.set_t(1) ** 2], [eps])

    assert sp.expand(steady_state_eq - (x.to_ss() ** 2 + x.to_ss() * (1 - rho))) == 0


@pytest.mark.parametrize(
    "error_function, expected",
    [("squared", 14.0), ("mean_squared", 14.0 / 3), ("abs", 6.0), ("l2-norm", np.sqrt(14.0))],
    ids=str,
)
def test_pt_error_from_resid(error_function, expected):
    resid = pt.dvector("resid")
    f = pytensor.function([resid], pt_error_from_resid(resid, error_function), mode="FAST_COMPILE")

    assert_allclose(f(np.array([1.0, -2.0, 3.0])), expected)


def test_pt_error_from_resid_rejects_unknown_function():
    with pytest.raises(NotImplementedError, match="Error function huber not implemented"):
        pt_error_from_resid(pt.dvector("resid"), "huber")


def test_compile_known_ss_symbolic_keys_only_known_variables():
    model = load_and_cache_model("rbc_2_block_partial_ss.gcn")
    _, cache = compile_param_dict_func(model._param_dict, model._deterministic_dict, return_symbolic=True)
    parameters = list(model._param_dict.to_sympy().keys()) + list(model._deterministic_dict.to_sympy().keys())

    mapping, _ = compile_known_ss(
        model._ss_solution_dict, model.variables, parameters, cache=cache, return_symbolic=True
    )

    known_names = [symbol.name for symbol in model._ss_solution_dict.to_sympy()]
    assert [node.name for node in mapping] == known_names


def test_wrong_user_solutions_raises():
    model_1 = load_and_cache_model("one_block_1.gcn")

    expected_msg = (
        "User-provided steady state is not valid. The following equations had non-zero residuals "
        "after substitution:\n(rho - 1)*log(A_ss)"
    )

    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        model_1.steady_state(fixed_values={"A_ss": 3.0})


def test_print_steady_state_report_solver_successful(caplog):
    model_1 = load_and_cache_model("one_block_1.gcn")
    res = model_1.steady_state(verbose=False, progressbar=False)

    expected_output = """A_ss               1.000
                         C_ss               4.119
                         K_ss              74.553
                         U_ss             101.458
                         lambda_ss          0.120"""

    print_steady_state(res)

    assert _collapse_whitespace(caplog.messages[-1]) == _collapse_whitespace(expected_output)


def test_print_steady_state_lists_calibrated_parameters_after_variables(caplog):
    model = load_and_cache_model("one_block_2_no_extra.gcn")
    res = model.steady_state(how="root", verbose=False, progressbar=False)

    expected_output = """A_ss               1.000
                         C_ss               0.360
                         I_ss               0.019
                         K_ss               0.975
                         L_ss               0.351
                         U_ss            -192.072
                         Y_ss               0.379
                         lambda_ss          1.887
                         q_ss               1.887


                         alpha              0.076"""

    print_steady_state(res)

    assert _collapse_whitespace(caplog.messages[-1]) == _collapse_whitespace(expected_output)


def test_print_steady_state_report_solver_fails(caplog):
    model_1 = load_and_cache_model("one_block_1.gcn")
    result = model_1.steady_state(verbose=False, progressbar=False)

    result.success = False
    print_steady_state(result)
    expected_output = """Values come from the latest solver iteration but are NOT a valid steady state.
                         A_ss               1.000
                         C_ss               4.119
                         K_ss              74.553
                         U_ss             101.458
                         lambda_ss          0.120"""

    assert _collapse_whitespace(caplog.messages[-1]) == _collapse_whitespace(expected_output)


@pytest.mark.parametrize(
    "fixed_values, expected_msg",
    [
        (
            {"K": 3.0, "K_ss": 3.0},
            "The following variables were provided twice (once with a _ss suffix and once without):\nK",
        ),
        (
            {"Z_ss": 3.0},
            "The following variables or calibrated parameters were given fixed steady state values but are unknown "
            "to the model: Z",
        ),
    ],
    ids=["duplicate_with_and_without_suffix", "unknown_variable"],
)
def test_invalid_fixed_values_raise(fixed_values, expected_msg):
    model_1 = load_and_cache_model("one_block_1.gcn")

    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        model_1.steady_state(fixed_values=fixed_values, verbose=False, progressbar=False)


def test_unknown_how_raises():
    model_1 = load_and_cache_model("one_block_1.gcn")

    with pytest.raises(NotImplementedError, match=re.escape("got 'newton'")):
        model_1.steady_state(how="newton")


def test_fixed_values_completing_analytic_steady_state_skip_the_solver():
    model = load_and_cache_model("one_block_1_ss.gcn")
    analytic = model.steady_state(verbose=False, progressbar=False)

    res = model.steady_state(fixed_values={"K_ss": analytic["K_ss"]}, verbose=False, progressbar=False)

    assert res.success
    assert res.to_string() == analytic.to_string()


def test_incomplete_ss_relationship_raises_with_root():
    model_1 = load_and_cache_model("one_block_1.gcn", infer_steady_state=False)
    expected_msg = (
        'Solving a partially provided steady state with how = "root" is only allowed if applying the given '
        "values results in a new square system.\n"
        "Remaining: 4 variables, 5 equations."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(expected_msg),
    ):
        model_1.steady_state(how="root", fixed_values={"K_ss": 3.0})


def test_wrong_and_incomplete_ss_relationship_fails_with_minimize():
    model_1 = load_and_cache_model("one_block_1.gcn", infer_steady_state=False)
    res = model_1.steady_state(verbose=False, progressbar=False, fixed_values={"K_ss": 3.0})
    assert not res.success


def test_numerical_solvers_succeed_and_agree():
    model_1 = load_and_cache_model("one_block_1.gcn")
    assert_root_and_minimize_agree(model_1)


@pytest.mark.parametrize(
    "how, kwargs",
    [
        ("root", {"use_jac": False}),
        ("root", {"jitter_x0": True}),
        ("minimize", {"jitter_x0": True}),
        ("minimize", {"use_hess": True, "use_hessp": False}),
    ],
    ids=["root_no_jac", "root_jitter", "minimize_jitter", "minimize_hess"],
)
def test_solver_options_reach_the_same_steady_state(how, kwargs):
    model_1 = load_and_cache_model("one_block_1.gcn")
    reference = model_1.steady_state(how="root", verbose=False, progressbar=False)

    res = model_1.steady_state(how=how, verbose=False, progressbar=False, **kwargs)

    assert res.success
    for k in reference:
        assert_allclose(res[k], reference[k], rtol=1e-6, err_msg=k)


def test_hess_and_hessp_together_warn_and_use_hessp(caplog):
    model_1 = load_and_cache_model("one_block_1.gcn")

    with caplog.at_level("WARNING"):
        res = model_1.steady_state(how="minimize", use_hess=True, use_hessp=True, verbose=False, progressbar=False)

    assert res.success
    assert "Both use_hess and use_hessp are set to True. use_hessp will be used." in caplog.messages


def test_steady_state_matches_analytic():
    model_1 = load_and_cache_model("one_block_1.gcn")
    param_dict = model_1.parameters().to_sympy()
    alpha, beta, delta, gamma, _rho = list(param_dict.keys())

    A_ss = sp.Float(1.0)
    K_ss = ((alpha * beta) / (1 - beta + beta * delta)) ** (1 / (1 - alpha))
    C_ss = K_ss**alpha - delta * K_ss
    lambda_ss = C_ss ** (-gamma)
    U_ss = 1 / (1 - beta) * (C_ss ** (1 - gamma) - 1) / (1 - gamma)

    ss_var = [x.to_ss().name for x in model_1.variables]
    ss_dict = {k: float(v.subs(param_dict)) for k, v in zip(ss_var, [A_ss, C_ss, K_ss, U_ss, lambda_ss], strict=True)}

    root_ss_dict = model_1.steady_state(verbose=False, progressbar=False, how="root")
    assert root_ss_dict.success

    minimize_ss_dict = model_1.steady_state(verbose=False, progressbar=False, how="minimize")
    assert minimize_ss_dict.success

    for param_name, ss_value in ss_dict.items():
        assert_allclose(ss_value, root_ss_dict[param_name])
        assert_allclose(ss_value, minimize_ss_dict[param_name])


def test_numerical_solvers_succeed_and_agree_w_calibrated_params():
    model = load_and_cache_model("one_block_2_no_extra.gcn")

    ss_root = model.steady_state(how="root", verbose=False, progressbar=False)
    assert ss_root.success

    # From the default x0 the minimizer can land in a different basin depending on the platform's floating point,
    # so seed it from the root solution and test that the two solvers agree there.
    x0 = np.array([float(ss_root[v.name]) for v in model._vars_to_solve])
    ss_minimize = model.steady_state(how="minimize", verbose=False, progressbar=False, optimizer_kwargs={"x0": x0})
    assert ss_minimize.success

    for k in ss_root:
        assert_allclose(ss_root[k], ss_minimize[k], err_msg=k)


@pytest.mark.parametrize(
    "assumptions, user_bound, expected_type",
    [
        ({"positive": True}, (0.0, 1.0), Interval),
        ({"unit_interval": True, "positive": True}, None, type(logodds)),
        ({"positive": True}, None, type(log)),
        ({"negative": True}, None, Interval),
        ({}, None, type(None)),
    ],
    ids=["explicit_bound_wins", "unit_interval", "positive", "negative", "unconstrained"],
)
def test_infer_variable_transform_waterfall(assumptions, user_bound, expected_type):
    variable = sp.Symbol("x", **assumptions)
    transform = infer_variable_transform(variable, user_bound=user_bound)
    assert isinstance(transform, expected_type)


@pytest.mark.parametrize(
    "assumptions, expected",
    [({"positive": True}, (1e-8, None)), ({"negative": True}, (None, -1e-8)), ({}, (None, None))],
    ids=["positive", "negative", "unconstrained"],
)
def test_infer_variable_bounds(assumptions, expected):
    assert infer_variable_bounds(TimeAwareSymbol("x", 0, **assumptions)) == expected


def test_transform_steady_state_system_round_trips_and_preserves_residuals():
    x, y = pt.dscalars("x", "y")
    equations = [x - 2.0, y + x]

    transformed, y_nodes, to_unconstrained, to_constrained = transform_steady_state_system(
        equations, [x, y], [log, None]
    )

    f_original = pytensor.function([x, y], equations, mode="FAST_COMPILE")
    f_transformed = pytensor.function(y_nodes, transformed, mode="FAST_COMPILE")

    constrained_point = np.array([3.0, -1.5])
    unconstrained_point = to_unconstrained(constrained_point)

    assert_allclose(unconstrained_point, [np.log(3.0), -1.5])
    assert_allclose(to_constrained(unconstrained_point), constrained_point)
    assert_allclose(f_transformed(*unconstrained_point), f_original(*constrained_point))


def test_prefer_transform_solves_unconstrained():
    model_1 = load_and_cache_model("one_block_1.gcn")
    ss_root = model_1.steady_state(how="root", verbose=False, progressbar=False)

    ss_transform = model_1.steady_state(how="minimize", prefer_transform=True, verbose=False, progressbar=False)
    assert ss_transform.success
    for k in ss_root:
        assert_allclose(ss_root[k], ss_transform[k], err_msg=k)


def test_steady_state_matches_analytic_w_calibrated_params():
    model_2 = load_and_cache_model(
        "one_block_2_no_extra.gcn",
        infer_steady_state=True,
    )
    param_dict = model_2.parameters().to_sympy()
    calib_params = model_2.calibrated_params

    beta, delta, _rho, tau, theta = list(param_dict.keys())
    (alpha,) = calib_params

    term_1 = theta * (1 - alpha) / (1 - theta)
    term_2 = alpha / (1 - beta + beta * delta)
    a_exp = alpha / (1 - alpha)

    A_ss = sp.Float(1.0)
    Y_ss = term_1 * term_2**a_exp / (1 + term_1 - delta * term_2)
    K_ss = term_2 * Y_ss
    L_ss = term_2 ** (-a_exp) * Y_ss
    C_ss = term_1 * term_2**a_exp - term_1 * Y_ss
    I_ss = delta * K_ss

    lambda_ss = theta * (C_ss**theta * (1 - L_ss) ** (1 - theta)) ** (1 - tau) / C_ss
    q_ss = lambda_ss

    U_ss = 1 / (1 - beta) * (C_ss**theta * (1 - L_ss) ** (1 - theta)) ** (1 - tau) / (1 - tau)

    f = sp.lambdify(alpha, (L_ss / K_ss - 0.36).simplify().subs(param_dict))
    res = optimize.root_scalar(f, bracket=[1e-4, 0.99])

    calib_solution = {alpha: res.root}
    all_params = param_dict | calib_solution

    answer_dict = {
        "A_ss": A_ss,
        "C_ss": C_ss,
        "I_ss": I_ss,
        "K_ss": K_ss,
        "L_ss": L_ss,
        "U_ss": U_ss,
        "Y_ss": Y_ss,
        "lambda_ss": lambda_ss,
        "q_ss": q_ss,
        "alpha": res.root,
    }

    numerical_ss_dict = model_2.steady_state(
        verbose=False,
        progressbar=False,
        how="minimize",
        use_jac=True,
        bounds={"alpha": (0.05, 0.7)},
        optimizer_kwargs={"method": "trust-constr", "options": {"maxiter": 50_000}},
    )
    assert numerical_ss_dict.success

    assert_allclose(numerical_ss_dict["L_ss"] / numerical_ss_dict["K_ss"], 0.36)

    ss_vars = [x.to_ss() for x in model_2.variables]
    for k in ss_vars:
        answer = float(answer_dict[k.name].subs(all_params))
        # trust-constr stops at its own gtol and xtol, so the numerical steady state matches the analytic one only to
        # about 1e-6.
        assert_allclose(answer, numerical_ss_dict[k.name], rtol=1e-6, err_msg=k.name)


def test_numerical_solvers_succeed_and_agree_RBC():
    model_3 = load_and_cache_model("rbc_2_block.gcn")
    assert_root_and_minimize_agree(model_3)


def test_RBC_steady_state_matches_analytic():
    model_3 = load_and_cache_model("rbc_2_block.gcn")
    param_dict = model_3.parameters().to_sympy()

    alpha, beta, delta, _rho_A, sigma_C, sigma_L = list(param_dict.keys())
    A_ss = sp.Float(1.0)
    r_ss = 1 / beta - (1 - delta)
    w_ss = (1 - alpha) * (alpha / r_ss) ** (alpha / (1 - alpha))
    Y_ss = (
        w_ss ** (1 / (sigma_L + sigma_C))
        * (w_ss / (1 - alpha)) ** (sigma_L / (sigma_L + sigma_C))
        * (r_ss / (r_ss - delta * alpha)) ** (sigma_C / (sigma_L + sigma_C))
    )

    C_ss = (w_ss) ** (1 / sigma_C) * (w_ss / (1 - alpha) / Y_ss) ** (sigma_L / sigma_C)

    lambda_ss = C_ss ** (-sigma_C)
    q_ss = lambda_ss
    I_ss = delta * alpha * Y_ss / r_ss
    K_ss = alpha * Y_ss / r_ss
    L_ss = (1 - alpha) * Y_ss / w_ss
    P_ss = (w_ss / (1 - alpha)) ** (1 - alpha) * (r_ss / alpha) ** alpha

    U_ss = 1 / (1 - beta) * (C_ss ** (1 - sigma_C) / (1 - sigma_C) - L_ss ** (1 + sigma_L) / (1 + sigma_L))

    TC_ss = -(r_ss * K_ss + w_ss * L_ss)

    answer_dict = {
        "A_ss": A_ss,
        "C_ss": C_ss,
        "I_ss": I_ss,
        "K_ss": K_ss,
        "L_ss": L_ss,
        "TC_ss": TC_ss,
        "U_ss": U_ss,
        "Y_ss": Y_ss,
        "lambda_ss": lambda_ss,
        "q_ss": q_ss,
        "r_ss": r_ss,
        "w_ss": w_ss,
    }

    numerical_ss_dict = model_3.steady_state(verbose=False, progressbar=False)
    ss_vars = [x.to_ss() for x in model_3.variables]

    for k in ss_vars:
        answer = float(answer_dict[k.name].subs(param_dict))
        assert_allclose(answer, numerical_ss_dict[k.name], err_msg=k.name)


@pytest.mark.include_nk
def test_numerical_solvers_succeed_and_agree_NK():
    model_4 = load_and_cache_model("full_nk_no_ss.gcn")

    # The solvers need the unit steady states of the shock and inflation processes pinned to converge. This leaves
    # mc_ss, r_G_ss, and r_ss to be found, which full_nk_partial_ss.gcn provides analytically.
    assert_root_and_minimize_agree(
        model_4,
        optimizer_kwargs={"maxiter": 50_000},
        fixed_values={
            "shock_technology_ss": 1.0,
            "shock_preference_ss": 1.0,
            "pi_ss": 1.0,
            "pi_star_ss": 1.0,
            "pi_obj_ss": 1.0,
        },
    )


@pytest.mark.include_nk
def test_steady_state_matches_analytic_NK():
    model_4 = load_and_cache_model("full_nk_no_ss.gcn")

    param_dict = model_4.parameters().to_sympy()
    (
        alpha,
        beta,
        delta,
        eta_p,
        eta_w,
        _gamma_I,
        _gamma_R,
        _gamma_Y,
        _gamma_pi,
        phi_H,
        _phi_pi_obj,
        psi_p,
        psi_w,
        _rho_pi_dot,
        _rho_preference,
        _rho_technology,
        sigma_C,
        sigma_L,
    ) = list(param_dict.keys())

    shock_technology_ss = sp.Float(1)
    shock_preference_ss = sp.Float(1)
    pi_ss = sp.Float(1)
    pi_star_ss = sp.Float(1)
    pi_obj_ss = sp.Float(1)

    r_ss = 1 / beta - (1 - delta)
    r_G_ss = 1 / beta

    mc_ss = 1 / (1 + psi_p)
    w_ss = (1 - alpha) * mc_ss ** (1 / (1 - alpha)) * (alpha / r_ss) ** (alpha / (1 - alpha))
    w_star_ss = w_ss

    Y_ss = (
        w_ss ** ((sigma_L + 1) / (sigma_C + sigma_L))
        * ((-beta * phi_H + 1) / (psi_w + 1)) ** (1 / (sigma_C + sigma_L))
        * (r_ss / ((1 - phi_H) * (-alpha * delta * mc_ss + r_ss))) ** (sigma_C / (sigma_C + sigma_L))
        / (mc_ss * (1 - alpha)) ** (sigma_L / (sigma_C + sigma_L))
    )

    C_ss = (
        w_ss ** ((1 + sigma_L) / sigma_C)
        * (1 / (1 - phi_H))
        * ((1 - beta * phi_H) / (1 + psi_w)) ** (1 / sigma_C)
        * ((1 - alpha) * mc_ss) ** (-sigma_L / sigma_C)
        * Y_ss ** (-sigma_L / sigma_C)
    )

    lambda_ss = (1 - beta * phi_H) * ((1 - phi_H) * C_ss) ** (-sigma_C)
    q_ss = lambda_ss
    I_ss = delta * alpha * mc_ss * Y_ss / r_ss
    K_ss = alpha * mc_ss * Y_ss / r_ss
    L_ss = (1 - alpha) * Y_ss * mc_ss / w_ss

    U_ss = (
        1 / (1 - beta) * (((1 - phi_H) * C_ss) ** (1 - sigma_C) / (1 - sigma_C) - L_ss ** (1 + sigma_L) / (1 + sigma_L))
    )

    TC_ss = -(r_ss * K_ss + w_ss * L_ss)
    Div_ss = Y_ss + TC_ss

    LHS_ss = 1 / (1 - beta * eta_p * pi_ss ** (1 / psi_p)) * lambda_ss * Y_ss * pi_star_ss

    RHS_ss = 1 / (1 + psi_p) * LHS_ss

    LHS_w_ss = 1 / (1 - beta * eta_w) * 1 / (1 + psi_w) * w_star_ss * lambda_ss * L_ss

    RHS_w_ss = LHS_w_ss

    answer_dict = {
        "C_ss": C_ss,
        "Div_ss": Div_ss,
        "I_ss": I_ss,
        "K_ss": K_ss,
        "LHS_ss": LHS_ss,
        "LHS_w_ss": LHS_w_ss,
        "L_ss": L_ss,
        "RHS_ss": RHS_ss,
        "RHS_w_ss": RHS_w_ss,
        "TC_ss": TC_ss,
        "U_ss": U_ss,
        "Y_ss": Y_ss,
        "lambda_ss": lambda_ss,
        "mc_ss": mc_ss,
        "pi_obj_ss": pi_obj_ss,
        "pi_star_ss": pi_star_ss,
        "pi_ss": pi_ss,
        "q_ss": q_ss,
        "r_G_ss": r_G_ss,
        "r_ss": r_ss,
        "shock_preference_ss": shock_preference_ss,
        "shock_technology_ss": shock_technology_ss,
        "w_star_ss": w_star_ss,
        "w_ss": w_ss,
    }

    numerical_ss_dict = model_4.steady_state(
        how="root",
        fixed_values={
            "shock_technology_ss": 1.0,
            "shock_preference_ss": 1.0,
            "pi_ss": 1.0,
            "pi_star_ss": 1.0,
            "pi_obj_ss": 1.0,
        },
        verbose=False,
        progressbar=False,
    )
    assert numerical_ss_dict.success

    ss_vars = [x.to_ss() for x in model_4.variables]
    for k in ss_vars:
        answer = float(answer_dict[k.name].subs(param_dict))
        assert_allclose(answer, numerical_ss_dict[k.name], err_msg=k.name)
