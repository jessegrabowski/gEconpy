import re

import numpy as np
import pytest
import sympy as sp

from numpy.testing import assert_allclose
from scipy import optimize

from gEconpy import model_from_gcn
from gEconpy.classes.containers import SymbolDictionary
from gEconpy.model.model import Model
from gEconpy.model.steady_state import solve as solve_module
from gEconpy.model.steady_state.solve import print_steady_state
from gEconpy.pytensorf.compile import compile_cache_info
from tests._resources.cache_compiled_models import load_and_cache_model
from tests.conftest import TEST_GCNS


def _collapse_whitespace(text: str) -> str:
    return re.sub(" +", " ", re.sub("[\t\n]", " ", text))


def assert_root_and_minimize_agree(model: Model, **steady_state_kwargs):
    ss_root = model.steady_state(how="root", verbose=False, progressbar=False, **steady_state_kwargs)
    ss_minimize = model.steady_state(how="minimize", verbose=False, progressbar=False, **steady_state_kwargs)

    assert ss_root.success
    assert ss_minimize.success

    for k in ss_root:
        assert_allclose(ss_root[k], ss_minimize[k], err_msg=k)


def test_wrong_user_solutions_raises():
    model_1 = load_and_cache_model("one_block_1.gcn")

    expected_msg = (
        "User-provided steady state is not valid. The following equations had non-zero residuals "
        "after substitution:\n(rho - 1)*log(A_ss)"
    )

    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        model_1.steady_state(fixed_values={"A_ss": 3.0})


LARGE_TERM_GCN = """
block MODEL
{
    identities
    {
        log(A[]) = rho * log(A[-1]) + epsilon[];
        X[] = A[] * w ^ 35;
    };

    shocks
    {
        epsilon[];
    };

    calibration
    {
        rho = 0.9;
        w = 1.75;
    };
};

block STEADY_STATE
{
    identities
    {
        A[ss] = 1;
        X[ss] = exp(35 * log(w));
    };
};
"""


@pytest.fixture
def large_term_model(tmp_path):
    """``X_ss`` is about 3e8, where float64 rounding alone leaves an absolute residual near 2e-8."""
    path = tmp_path / "large_term.gcn"
    path.write_text(LARGE_TERM_GCN)
    return model_from_gcn(path, verbose=False)


def test_exact_steady_state_with_large_terms_succeeds(large_term_model):
    result = large_term_model.steady_state(verbose=False, progressbar=False)

    assert result.success
    assert_allclose(result["X_ss"], 1.75**35)


def test_an_infinite_residual_is_never_within_tolerance():
    assert not solve_module._residual_within_tolerance(np.array([np.inf]), scale=np.array([np.inf])).any()


def test_large_terms_do_not_hide_a_wrong_fixed_value(large_term_model):
    result = large_term_model.steady_state(
        fixed_values={"X_ss": 1.75**35 * (1 + 1e-6)}, verbose=False, progressbar=False
    )

    assert not result.success


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


def test_fixed_values_completing_analytic_steady_state_skip_the_solver(monkeypatch):
    model = load_and_cache_model("one_block_1_ss.gcn")
    analytic = model.steady_state(verbose=False, progressbar=False)

    def no_solve(**_):
        raise AssertionError("the steady state is fully determined, so no solver should run")

    monkeypatch.setattr(solve_module, "root", no_solve)
    monkeypatch.setattr(solve_module, "minimize", no_solve)
    res = model.steady_state(fixed_values={"K_ss": analytic["K_ss"]}, verbose=False, progressbar=False)

    assert res.success
    assert res.keys() == analytic.keys()
    assert_allclose([res[key] for key in analytic], list(analytic.values()), rtol=1e-14)


def test_fixed_value_that_squares_the_system_solves_with_root():
    """Fixing ``A_ss`` turns the technology equation into a check row, which leaves a square system for root."""
    model = model_from_gcn(TEST_GCNS / "one_block_1.gcn", verbose=False, infer_steady_state=False)
    reference = model.steady_state(how="minimize", verbose=False, progressbar=False)

    res = model.steady_state(how="root", fixed_values={"A_ss": 1.0}, verbose=False, progressbar=False)

    assert res.success
    assert_allclose([res[key] for key in reference], list(reference.values()), rtol=1e-8)


def test_violated_check_row_fails_the_verdict_and_names_the_equation(caplog):
    """A fixed value inconsistent with a known one breaks a row the solve never sees, and the log has to say so."""
    k_ss = load_and_cache_model("one_block_1_ss.gcn").steady_state(verbose=False, progressbar=False)["K_ss"]

    # The known values are read on the first solve, so they are dropped before the model ever solves.
    model = model_from_gcn(TEST_GCNS / "one_block_1_ss.gcn", verbose=False)
    known = model._ss_solution_dict.to_sympy()
    model._ss_solution_dict = SymbolDictionary({key: value for key, value in known.items() if key.base_name != "U"})

    with caplog.at_level("WARNING"):
        res = model.steady_state(fixed_values={"K_ss": 1.1 * k_ss}, verbose=True, progressbar=False)

    assert not res.success
    assert any("equations the solve does not involve are violated" in message for message in caplog.messages)
    assert any("in Household.foc.L" in message for message in caplog.messages)


def test_repeat_solves_at_new_parameters_compile_nothing():
    model = model_from_gcn(TEST_GCNS / "one_block_1.gcn", verbose=False)
    model.steady_state(how="root", verbose=False, progressbar=False)
    # A recompile of a structurally identical graph is a cache hit, so hits and misses are both counted.
    compiles = compile_cache_info().hits + compile_cache_info().misses

    for beta in [0.97, 0.98]:
        assert model.steady_state(how="root", beta=beta, verbose=False, progressbar=False).success

    assert compile_cache_info().hits + compile_cache_info().misses == compiles


@pytest.mark.parametrize("bound", [(1e-8, None), [1e-8, None]], ids=["tuple", "list"])
def test_bounds_reparametrize_the_minimize_solve_whether_tuple_or_list(bound):
    model = load_and_cache_model("one_block_1.gcn")

    res = model.steady_state(how="minimize", bounds={"lambda_ss": bound}, verbose=False, progressbar=False)

    assert res.success
    assert res["lambda_ss"] > 0


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


@pytest.mark.parametrize(
    "optimizer_kwargs, method",
    [(None, "trust-ncg"), ({"method": "newton-cg"}, "newton-cg"), ({"method": "dogleg"}, "dogleg")],
    ids=["default_method", "newton-cg", "dogleg"],
)
def test_minimize_without_jac_rejects_gradient_methods(optimizer_kwargs, method):
    model_1 = load_and_cache_model("one_block_1.gcn")

    with pytest.raises(ValueError, match=re.escape(f"Method {method!r} requires a gradient")):
        model_1.steady_state(
            how="minimize",
            use_jac=False,
            use_hessp=False,
            verbose=False,
            progressbar=False,
            optimizer_kwargs=optimizer_kwargs,
        )


def test_minimize_without_jac_accepts_gradient_method_when_hessp_builds_the_gradient():
    model_1 = load_and_cache_model("one_block_1.gcn")
    reference = model_1.steady_state(verbose=False, progressbar=False)

    result = model_1.steady_state(how="minimize", use_jac=False, use_hessp=True, verbose=False, progressbar=False)

    assert result.success
    for name, value in reference.items():
        assert_allclose(result[name], value, rtol=1e-6, err_msg=name)


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


def test_solver_does_not_pass_inferred_bounds_to_the_transform(monkeypatch):
    """
    The inferred box bounds guard the sign for box-constrained methods only.

    Passing them as an explicit bound made every positive variable look bounded, so a ``unit_interval`` variable
    was reparametrized under a shifted log instead of a logit.
    """
    captured: list = []
    original = solve_module.transform_steady_state_system

    def record(equations, ss_nodes, transforms):
        captured.extend(transforms)
        return original(equations, ss_nodes, transforms)

    monkeypatch.setattr(solve_module, "transform_steady_state_system", record)

    # A cached model may already hold a compiled solver, which would skip the transform choice.
    model = model_from_gcn(TEST_GCNS / "one_block_2_no_extra.gcn", verbose=False)
    model.steady_state(how="minimize", verbose=False, progressbar=False)

    chosen = dict(zip(model._vars_to_solve, captured, strict=True))
    alpha_transform = next(t for v, t in chosen.items() if getattr(v, "base_name", v.name) == "alpha")

    assert float(alpha_transform.backward(0.0).eval()) == pytest.approx(0.5)


def test_declared_bounds_choose_the_transform(monkeypatch):
    """
    A declared bound reaches the solver and outranks the assumption it implies.

    ``alpha`` is declared on ``(0.01, 0.99)``, which implies ``positive``. Were the bound not reaching the solver,
    that assumption would select a log transform, whose backward maps 0.0 to 1.0 rather than to the interval's
    midpoint.
    """
    captured: list = []
    original = solve_module.transform_steady_state_system

    def record(equations, ss_nodes, transforms):
        captured.extend(transforms)
        return original(equations, ss_nodes, transforms)

    monkeypatch.setattr(solve_module, "transform_steady_state_system", record)

    model = model_from_gcn(TEST_GCNS / "one_block_2_symbols.gcn", verbose=False)
    alpha = next(v for v in model._vars_to_solve if getattr(v, "base_name", v.name) == "alpha")

    model.steady_state(how="minimize", verbose=False, progressbar=False)

    chosen = dict(zip(model._vars_to_solve, captured, strict=True))
    assert float(chosen[alpha].backward(0.0).eval()) == pytest.approx(0.5)


def test_grad_tol_never_reaches_the_optimizer():
    """A ``grad_tol`` leaked into ``optimizer_kwargs`` raises in scipy, which the wrapper reports as a failed solve."""
    model = load_and_cache_model("one_block_1.gcn")

    assert model.steady_state(how="minimize", grad_tol=1e-2, verbose=False, progressbar=False).success


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

    # From the default x0 the minimizer can land in a different basin depending on the platform's floating
    # point, and alpha solves close to its lower bound, so seed it from the root solution.
    root_ss_dict = model_2.steady_state(how="root", verbose=False, progressbar=False)
    assert root_ss_dict.success

    x0 = np.array([float(root_ss_dict[v.name]) for v in model_2._vars_to_solve])
    numerical_ss_dict = model_2.steady_state(
        verbose=False,
        progressbar=False,
        how="minimize",
        use_jac=True,
        bounds={"alpha": (0.05, 0.7)},
        optimizer_kwargs={"method": "trust-constr", "options": {"maxiter": 50_000}, "x0": x0},
    )
    assert numerical_ss_dict.success

    assert_allclose(numerical_ss_dict["L_ss"] / numerical_ss_dict["K_ss"], 0.36)

    # The loop below walks model variables, and alpha is a calibrated parameter, so it needs its own assertion.
    assert_allclose(numerical_ss_dict["alpha"], answer_dict["alpha"])

    ss_vars = [x.to_ss() for x in model_2.variables]
    for k in ss_vars:
        answer = float(answer_dict[k.name].subs(all_params))
        # trust-constr stops on its own xtol rather than at a root, so the numerical steady state matches the
        # analytic one to about 1e-9 here and the tolerance leaves room for the platform's arithmetic.
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


@pytest.mark.filterwarnings("ignore:Numba will use object mode:UserWarning")
def test_a_root_call_solves_its_variable_inside_the_known_steady_state():
    model = model_from_gcn(TEST_GCNS / "tank_informal_root.gcn", verbose=False)

    result = model.steady_state(verbose=False, progressbar=False)
    known = model.f_ss(**model.parameters())

    assert result.success
    # The informal wage at which informal labor demand meets hand-to-mouth supply, from a one-dimensional brentq solve.
    assert_allclose(result["w_I_ss"], 0.679473, rtol=1e-6)
    assert_allclose(known["w_I_ss"], result["w_I_ss"])
