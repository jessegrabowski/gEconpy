import re

import pytest
import sympy as sp

from numpy.testing import assert_allclose
from scipy import optimize

from gEconpy import model_from_gcn
from gEconpy.model.model import Model
from gEconpy.model.steady_state import print_steady_state
from tests.test_model_loaders import JAX_INSTALLED
from tests.utilities.shared_fixtures import load_and_cache_model


def root_and_min_agree_helper(model: Model, **kwargs):
    verbose = kwargs.pop("verbose", False)
    progressbar = kwargs.pop("progressbar", True)
    root_method = kwargs.pop("root_method", None)
    minimize_method = kwargs.pop("minimize_method", None)
    optimizer_kwargs = kwargs.pop("optimizer_kwargs", {})

    _ = kwargs.pop("how", None)

    if root_method:
        optimizer_kwargs["method"] = root_method

    ss_root, root_success = model.steady_state(
        how="root",
        verbose=verbose,
        progressbar=progressbar,
        optimizer_kwargs=optimizer_kwargs,
        **kwargs,
    )

    if minimize_method:
        optimizer_kwargs["method"] = minimize_method
    ss_minimize, minimize_success = model.steady_state(
        how="minimize",
        verbose=verbose,
        progressbar=progressbar,
        optimizer_kwargs=optimizer_kwargs,
        **kwargs,
    )

    assert root_success
    assert minimize_success

    for k in ss_root.keys():
        assert_allclose(ss_root[k], ss_minimize[k], err_msg=k)


def test_solve_ss_with_partial_user_solution():
    model_1 = load_and_cache_model(
        "one_block_1.gcn", backend="numpy", use_jax=JAX_INSTALLED
    )
    res, success = model_1.steady_state()
    assert success


def test_wrong_user_solutions_raises():
    model_1 = load_and_cache_model(
        "one_block_1.gcn", backend="numpy", use_jax=JAX_INSTALLED
    )

    expected_msg = (
        "User-provide steady state is not valid. The following equations had non-zero residuals "
        "after subsitution:\n(rho - 1)*log(A_ss)"
    )

    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        model_1.steady_state(fixed_values={"A_ss": 3.0})


def test_print_steady_state_report_solver_successful(caplog):
    model_1 = load_and_cache_model(
        "one_block_1.gcn", backend="numpy", use_jax=JAX_INSTALLED
    )
    res, success = model_1.steady_state(verbose=False, progressbar=False)

    expected_output = """A_ss               1.000
                         C_ss               4.119
                         K_ss              74.553
                         U_ss             101.458
                         lambda_ss          0.120"""

    expected_output = re.sub("[\t\n]", " ", expected_output)
    expected_output = re.sub(" +", " ", expected_output)

    print_steady_state(res, success)
    emitted_message = caplog.messages[-1]

    emitted_message = re.sub("[\t\n]", " ", emitted_message)
    emitted_message = re.sub(" +", " ", emitted_message)

    assert emitted_message == expected_output


def test_print_steady_state_report_solver_fails(caplog):
    model_1 = load_and_cache_model(
        "one_block_1.gcn", backend="numpy", use_jax=JAX_INSTALLED
    )
    result, _ = model_1.steady_state(verbose=False, progressbar=False)

    # Spoof a failed solving attempt
    success = False
    print_steady_state(result, success)
    expected_output = """Values come from the latest solver iteration but are NOT a valid steady state.
                         A_ss               1.000
                         C_ss               4.119
                         K_ss              74.553
                         U_ss             101.458
                         lambda_ss          0.120"""
    expected_output = re.sub("[\t\n]", " ", expected_output)
    expected_output = re.sub(" +", " ", expected_output)

    emitted_message = caplog.messages[-1]
    emitted_message = re.sub("[\t\n]", " ", emitted_message)
    emitted_message = re.sub(" +", " ", emitted_message)

    assert emitted_message == expected_output


def test_incomplete_ss_relationship_raises_with_root():
    model_1 = load_and_cache_model(
        "one_block_1.gcn", backend="numpy", use_jax=JAX_INSTALLED
    )
    expected_msg = (
        'Solving a partially provided steady state with how = "root" is only allowed if applying the given '
        "values results in a new square system.\n"
        "Found: 1 provided steady state value\n"
        "Eliminated: 0 equations."
    )
    with pytest.raises(
        ValueError,
        match=expected_msg,
    ):
        model_1.steady_state(how="root", fixed_values={"K_ss": 3.0})


def test_wrong_and_incomplete_ss_relationship_fails_with_minimize():
    model_1 = load_and_cache_model(
        "one_block_1.gcn", backend="numpy", use_jax=JAX_INSTALLED
    )
    res, success = model_1.steady_state(
        verbose=False, progressbar=False, fixed_values={"K_ss": 3.0}
    )
    assert not success


def test_numerical_solvers_suceed_and_agree():
    model_1 = load_and_cache_model(
        "one_block_1.gcn", backend="numpy", use_jax=JAX_INSTALLED
    )
    root_and_min_agree_helper(model_1)


def test_steady_state_matches_analytic():
    model_1 = load_and_cache_model(
        "one_block_1.gcn", backend="numpy", use_jax=JAX_INSTALLED
    )
    param_dict = model_1.parameters().to_sympy()
    alpha, beta, delta, gamma, rho = list(param_dict.keys())

    A_ss = sp.Float(1.0)
    K_ss = ((alpha * beta) / (1 - beta + beta * delta)) ** (1 / (1 - alpha))
    C_ss = K_ss**alpha - delta * K_ss
    lambda_ss = C_ss ** (-gamma)
    U_ss = 1 / (1 - beta) * (C_ss ** (1 - gamma) - 1) / (1 - gamma)

    ss_var = [x.to_ss().name for x in model_1.variables]
    ss_dict = {
        k: float(v.subs(param_dict))
        for k, v in zip(ss_var, [A_ss, C_ss, K_ss, U_ss, lambda_ss])
    }

    root_ss_dict, success = model_1.steady_state(
        verbose=False, progressbar=False, how="root"
    )
    assert success

    minimize_ss_dict, success = model_1.steady_state(
        verbose=False, progressbar=False, how="minimize"
    )
    assert success

    for k in ss_dict:
        assert_allclose(ss_dict[k], root_ss_dict[k])
        assert_allclose(ss_dict[k], minimize_ss_dict[k])


def test_numerical_solvers_succeed_and_agree_w_calibrated_params():
    model_2 = load_and_cache_model(
        "one_block_2_no_extra.gcn",
        backend="numpy",
        use_jax=JAX_INSTALLED,
    )
    root_and_min_agree_helper(model_2)


def test_steady_state_matches_analytic_w_calibrated_params():
    model_2 = load_and_cache_model(
        "one_block_2_no_extra.gcn",
        backend="numpy",
        use_jax=JAX_INSTALLED,
    )
    param_dict = model_2.parameters().to_sympy()
    calib_params = model_2.calibrated_params

    beta, delta, rho, tau, theta = list(param_dict.keys())
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

    U_ss = (
        1
        / (1 - beta)
        * (C_ss**theta * (1 - L_ss) ** (1 - theta)) ** (1 - tau)
        / (1 - tau)
    )

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

    numerical_ss_dict, success = model_2.steady_state(verbose=False, progressbar=False)
    assert success

    # Test calibration of alpha --> L_ss / K_ss = 0.36
    assert_allclose(numerical_ss_dict["L_ss"] / numerical_ss_dict["K_ss"], 0.36)

    ss_vars = [x.to_ss() for x in model_2.variables]
    for k in ss_vars:
        answer = float(answer_dict[k.name].subs(all_params))
        assert_allclose(answer, numerical_ss_dict[k.name], err_msg=k.name)


def test_numerical_solvers_succeed_and_agree_RBC():
    model_3 = load_and_cache_model(
        "rbc_2_block.gcn", backend="numpy", use_jax=JAX_INSTALLED
    )
    root_and_min_agree_helper(model_3)


def test_RBC_steady_state_matches_analytic():
    model_3 = load_and_cache_model(
        "rbc_2_block.gcn", backend="numpy", use_jax=JAX_INSTALLED
    )
    param_dict = model_3.parameters().to_sympy()

    alpha, beta, delta, rho_A, sigma_C, sigma_L = list(param_dict.keys())
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

    U_ss = (
        1
        / (1 - beta)
        * (
            C_ss ** (1 - sigma_C) / (1 - sigma_C)
            - L_ss ** (1 + sigma_L) / (1 + sigma_L)
        )
    )

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

    numerical_ss_dict, success = model_3.steady_state(verbose=False, progressbar=False)
    ss_vars = [x.to_ss() for x in model_3.variables]

    for k in ss_vars:
        answer = float(answer_dict[k.name].subs(param_dict))
        assert_allclose(answer, numerical_ss_dict[k.name], err_msg=k.name)


def test_numerical_solvers_succeed_and_agree_NK():
    model_4 = load_and_cache_model(
        "full_nk_no_ss.gcn", backend="numpy", use_jax=JAX_INSTALLED
    )

    # This model's SS can't be solved without some help, so we provide the "obvious" solutions
    # This is almost equivalent to the full_nk_partial_ss.gcn, with a bit less info
    # (No solution for mc_ss, r_G, and r)
    root_and_min_agree_helper(
        model_4,
        verbose=False,
        progressbar=False,
        optimizer_kwargs={"maxiter": 50_000},
        fixed_values={
            "shock_technology_ss": 1.0,
            "shock_preference_ss": 1.0,
            "pi_ss": 1.0,
            "pi_star_ss": 1.0,
            "pi_obj_ss": 1.0,
        },
    )


def test_steady_state_matches_analytic_NK():
    model_4 = load_and_cache_model(
        "full_nk_no_ss.gcn", backend="numpy", use_jax=JAX_INSTALLED
    )

    param_dict = model_4.parameters().to_sympy()
    (
        alpha,
        beta,
        delta,
        eta_p,
        eta_w,
        gamma_I,
        gamma_R,
        gamma_Y,
        gamma_pi,
        phi_H,
        phi_pi_obj,
        psi_p,
        psi_w,
        rho_pi_dot,
        rho_preference,
        rho_technology,
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
    w_ss = (
        (1 - alpha)
        * mc_ss ** (1 / (1 - alpha))
        * (alpha / r_ss) ** (alpha / (1 - alpha))
    )
    w_star_ss = w_ss

    Y_ss = (
        w_ss ** ((sigma_L + 1) / (sigma_C + sigma_L))
        * ((-beta * phi_H + 1) / (psi_w + 1)) ** (1 / (sigma_C + sigma_L))
        * (r_ss / ((1 - phi_H) * (-alpha * delta * mc_ss + r_ss)))
        ** (sigma_C / (sigma_C + sigma_L))
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
        1
        / (1 - beta)
        * (
            ((1 - phi_H) * C_ss) ** (1 - sigma_C) / (1 - sigma_C)
            - L_ss ** (1 + sigma_L) / (1 + sigma_L)
        )
    )

    TC_ss = -(r_ss * K_ss + w_ss * L_ss)
    Div_ss = Y_ss + TC_ss

    LHS_ss = (
        1 / (1 - beta * eta_p * pi_ss ** (1 / psi_p)) * lambda_ss * Y_ss * pi_star_ss
    )

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

    numerical_ss_dict, success = model_4.steady_state(
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
    assert success

    ss_vars = [x.to_ss() for x in model_4.variables]
    for k in ss_vars:
        answer = float(answer_dict[k.name].subs(param_dict))
        assert_allclose(answer, numerical_ss_dict[k.name], err_msg=k.name)
