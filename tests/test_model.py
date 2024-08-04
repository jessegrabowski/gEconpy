import os
import unittest

from unittest import mock

import numdifftools as nd
import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_allclose

from gEconpy.exceptions.exceptions import GensysFailedException
from gEconpy.model.build import model_from_gcn
from gEconpy.model.compile import BACKENDS
from gEconpy.model.model import scipy_wrapper
from tests.utilities.expected_matrices import expected_linearization_result
from tests.utilities.shared_fixtures import load_and_cache_model


@pytest.fixture
def gcn_file_1():
    GCN_file = """
                block HOUSEHOLD
                {
                    definitions
                    {
                        u[] = log(C[]);
                    };

                    objective
                    {
                        U[] = u[] + beta * E[][U[1]];
                    };

                    controls
                    {
                        C[], K[];
                    };

                    constraints
                    {
                        Y[] = K[-1] ^ alpha;
                        C[] = r[] * K[-1];
                        K[] = (1 - delta) * K[-1];
                        X[] = Y[] + C[];
                        Z[] = 3;
                    };

                    calibration
                    {
                        alpha = 0.33;
                        beta = 0.99;
                        delta = 0.035;
                    };
                };
                """
    return GCN_file


expected_warnings = [
    "Simplification via a tryreduce block was requested but not possible because the system is not well defined.",
    "Removal of constant variables was requested but not possible because the system is not well defined.",
    "The model does not appear correctly specified, there are 8 equations but 12 variables. It will not be possible to "
    "solve this model. Please check the specification using available diagnostic tools, and check the GCN file for "
    "typos.",
]


@pytest.mark.parametrize(
    ["simplify_tryreduce", "simplify_constants", "expected_warning"],
    [
        (True, False, expected_warnings[0]),
        (False, True, expected_warnings[1]),
        (False, False, expected_warnings[2]),
    ],
    ids=["tryreduce", "constants", "no_simplify"],
)
def test_build_warns_if_model_not_defined(
    gcn_file_1, simplify_tryreduce, simplify_constants, expected_warning
):
    with unittest.mock.patch(
        "builtins.open",
        new=unittest.mock.mock_open(read_data=gcn_file_1),
        create=True,
    ):
        with pytest.warns(UserWarning, match=expected_warning):
            model_from_gcn(
                gcn_file_1,
                simplify_constants=simplify_constants,
                simplify_tryreduce=simplify_tryreduce,
                verbose=not (simplify_tryreduce or simplify_constants),
            )


simple_vars = ["L", "K", "A", "Y", "I", "C", "q", "U", "lambda", "q"]
simple_params = ["alpha", "theta", "beta", "delta", "tau", "rho"]
simple_shocks = ["epsilon"]

open_vars = [
    "A",
    "IIP",
    "r",
    "r_given",
    "KtoN",
    "N",
    "K",
    "C",
    "U",
    "Y",
    "I",
    "TB",
    "TBtoY",
    "CA",
    "lambda",
]
open_params = [
    "beta",
    "delta",
    "gamma",
    "omega",
    "psi2",
    "psi",
    "alpha",
    "rstar",
    "IIPbar",
    "rho_A",
]
open_shocks = ["epsilon_A"]

nk_vars = [
    "shock_technology",
    "shock_preference",
    "pi",
    "pi_star",
    "pi_obj",
    "B",
    "r",
    "r_G",
    "mc",
    "w",
    "w_star",
    "Y",
    "C",
    "lambda",
    "q",
    "I",
    "K",
    "L",
    "U",
    "TC",
    "Div",
    "LHS",
    "RHS",
    "LHS_w",
    "RHS_w",
]
nk_params = [
    "delta",
    "beta",
    "sigma_C",
    "sigma_L",
    "gamma_I",
    "phi_H",
    "psi_w",
    "eta_w",
    "alpha",
    "rho_technology",
    "rho_preference",
    "psi_p",
    "eta_p",
    "gamma_R",
    "gamma_pi",
    "gamma_Y",
    "phi_pi_obj",
    "rho_pi_dot",
]
nk_shocks = ["epsilon_R", "epsilon_pi", "epsilon_Y", "epsilon_preference"]


@pytest.mark.parametrize(
    "gcn_path, expected_variables, expected_params, expected_shocks",
    [
        (
            "One_Block_Simple_1_w_Steady_State.gcn",
            simple_vars,
            simple_params,
            simple_shocks,
        ),
        ("Open_RBC.gcn", open_vars, open_params, open_shocks),
        ("Full_New_Keynesian.gcn", nk_vars, nk_params, nk_shocks),
    ],
)
def test_variables_parsed(
    gcn_path, expected_variables, expected_params, expected_shocks
):
    file_path = os.path.join("tests/Test GCNs", gcn_path)
    model = model_from_gcn(
        file_path,
        verbose=False,
        backend="numpy",
        mode="FAST_COMPILE",
        simplify_constants=False,
        simplify_tryreduce=False,
    )

    model_vars = [v.base_name for v in model.variables]
    model_params = [
        p.name
        for p in model.params + model.calibrated_params + model.deterministic_params
    ]
    model_shocks = [s.base_name for s in model.shocks]

    assert (
        set(model_vars) - set(expected_variables) == set()
        and set(expected_variables) - set(model_vars) == set()
    )
    assert (
        set(model_params) - set(expected_params) == set()
        and set(expected_params) - set(model_params) == set()
    )
    assert (
        set(model_shocks) - set(expected_shocks) == set()
        and set(expected_shocks) - set(model_shocks) == set()
    )


@pytest.mark.parametrize(
    "gcn_path, name",
    [
        ("One_Block_Simple_1_w_Distributions.gcn", "one_block_prior"),
        ("One_Block_Simple_1_w_Steady_State.gcn", "one_block_ss"),
        ("Full_New_Keynesian.gcn", "full_nk"),
    ],
    ids=["one_block_prior", "one_block_ss", "full_nk"],
)
@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
def test_model_parameters(
    load_and_cache_model, gcn_path: str, name: str, backend: BACKENDS
):
    model = load_and_cache_model(gcn_path, backend)

    # Test default parameters
    params = model.parameters()

    assert all([params[k] == model._default_params[k] for k in model._default_params])
    assert all(isinstance(v, float) for v in params.values())

    # Test parameter update
    old_params = model._default_params.copy()
    params = model.parameters(beta=0.5)
    assert params["beta"] == 0.5
    assert model._default_params["beta"] == old_params["beta"]


@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
def test_deterministic_model_parameters(load_and_cache_model, backend: BACKENDS):
    model = load_and_cache_model("One_Block_Simple_2.gcn", backend)
    params = model.parameters()

    # Test numeric expression in calibration block
    assert_allclose(params["beta"], 1 / 1.01)

    # Test deterministic relationship
    params = model.parameters(theta=0.9)
    assert params["theta"] == 0.9
    assert_allclose(params["zeta"], -np.log(0.9))


@pytest.mark.parametrize(
    "gcn_path",
    ["One_Block_Simple_1_w_Steady_State.gcn", "Open_RBC.gcn", "Full_New_Keynesian.gcn"],
    ids=["one_block_prior", "one_block_ss", "full_nk"],
)
def test_all_backends_agree_on_parameters(load_and_cache_model, gcn_path):
    models = [
        load_and_cache_model(gcn_path, backend)
        for backend in ["numpy", "numba", "pytensor"]
    ]
    params = [np.r_[list(model.parameters().values())] for model in models]

    for i in range(3):
        for j in range(i):
            assert_allclose(params[i], params[j])


@pytest.mark.parametrize(
    "gcn_path",
    ["One_Block_Simple_1_w_Steady_State.gcn", "Open_RBC.gcn", "Full_New_Keynesian.gcn"],
    ids=["one_block_prior", "one_block_ss", "full_nk"],
)
@pytest.mark.parametrize(
    "func",
    ["f_ss_error_grad", "f_ss_error_hess", "f_ss_jac"],
    ids=["grad", "hess", "jac"],
)
def test_all_backends_agree_on_functions(load_and_cache_model, gcn_path, func):
    backends = ["numpy", "numba", "pytensor"]
    models = [load_and_cache_model(gcn_path, backend) for backend in backends]
    params = models[0].parameters().to_string()
    ss_vars = [x.to_ss().name for x in models[0].variables]
    x0 = dict(zip(ss_vars, np.full(len(models[0].variables), 0.8)))

    vals = [getattr(model, func)(**params, **x0) for model in models]
    for i in range(3):
        for j in range(i):
            assert_allclose(
                vals[i], vals[j], err_msg=f"{backends[i]} and {backends[j]} disagree"
            )


@pytest.mark.parametrize(
    "gcn_path",
    [
        "Two_Block_RBC_w_Partial_Steady_State.gcn",
        "Full_New_Keynesian_w_Partial_Steady_state.gcn",
    ],
    ids=["two_block", "full_nk"],
)
@pytest.mark.parametrize(
    "func", ["f_ss_error_grad", "f_ss_error_hess"], ids=["grad", "hess"]
)
def test_scipy_wrapped_functions_agree(load_and_cache_model, gcn_path, func):
    backend_names = ["numpy", "numba", "pytensor"]
    models = [load_and_cache_model(gcn_path, backend) for backend in backend_names]

    ss_variables = [x.to_ss() for x in models[0].variables]
    known_variables = list(models[0].f_ss(**models[0].parameters()).to_sympy().keys())
    vars_to_solve = [var for var in ss_variables if var not in known_variables]
    var_names_to_solve = [x.name for x in vars_to_solve]
    unknown_var_idx = np.array([x in vars_to_solve for x in ss_variables], dtype="bool")
    params = models[0].parameters().to_string()
    x0 = np.full(len(var_names_to_solve), 0.8)

    vals = [
        scipy_wrapper(
            getattr(model, func), var_names_to_solve, unknown_var_idx, model.f_ss
        )(x0, params)
        for model in models
    ]
    for i in range(3):
        for j in range(i):
            assert_allclose(
                vals[i],
                vals[j],
                err_msg=f"{backend_names[i]} and {backend_names[j]} disagree",
                rtol=1e-8,
                atol=1e-8,
            )


@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
@pytest.mark.parametrize(
    ("gcn_file", "expected_result"),
    [
        (
            "One_Block_Simple_1_w_Steady_State.gcn",
            {
                "A_ss": 1.0,
                "C_ss": 0.91982617,
                "I_ss": 0.27872301,
                "K_ss": 13.9361507,
                "L_ss": 0.3198395,
                "U_ss": -132.00424906,
                "Y_ss": 1.19854918,
                "lambda_ss": 0.51233068,
                "q_ss": 0.51233068,
            },
        ),
        (
            "Open_RBC.gcn",
            {
                "A_ss": 1.00000000e00,
                "CA_ss": 0.00000000e00,
                "C_ss": 9.23561040e00,
                "IIP_ss": 0.00000000e00,
                "I_ss": 2.73647613e00,
                "K_ss": 1.09459045e02,
                "KtoN_ss": 2.59033302e01,
                "N_ss": 4.22567464e00,
                "TB_ss": 0.00000000e00,
                "TBtoY_ss": 0.00000000e00,
                "U_ss": 7.32557872e01,
                "Y_ss": 1.19720865e01,
                "lambda_ss": 7.54570414e-02,
                "r_ss": 1.00000101e-02,
                "r_given_ss": 1.00000101e-02,
            },
        ),
        (
            "Full_New_Keynesian.gcn",
            {
                "C_ss": 1.50620761e00,
                "Div_ss": 6.69069052e-01,
                "I_ss": 2.77976530e-01,
                "K_ss": 1.11190612e01,
                "LHS_ss": 6.16941715e00,
                "LHS_w_ss": 1.40646786e00,
                "L_ss": 6.66135866e-01,
                "RHS_ss": 3.85588572e00,
                "RHS_w_ss": 1.40646786e00,
                "TC_ss": -1.11511509e00,
                "U_ss": -1.47270439e02,
                "Y_ss": 1.78418414e00,
                "q_ss": 8.90392916e-01,
                "mc_ss": 6.25000000e-01,
                "shock_preference_ss": 1.00000000e00,
                "shock_technology_ss": 1.00000000e00,
                "pi_ss": 1.00000000e00,
                "lambda_ss": 8.90392916e-01,
                "r_G_ss": 1.01010101e00,
                "r_ss": 3.51010101e-02,
                "pi_obj_ss": 1.00000000e00,
                "pi_star_ss": 1.00000000e00,
                "w_ss": 1.08810356e00,
                "w_star_ss": 1.08810356e00,
            },
        ),
    ],
    ids=["one_block", "open_rbc", "nk"],
)
def test_steady_state(
    load_and_cache_model, backend: BACKENDS, gcn_file: str, expected_result: np.ndarray
):
    n = len(expected_result)

    model = load_and_cache_model(gcn_file, backend)

    params = model.parameters()
    ss_dict = model.f_ss(**params)
    ss = np.array(np.r_[list(ss_dict.values())])
    expected_ss = np.r_[[expected_result[var] for var in ss_dict.to_string().keys()]]

    assert_allclose(ss, expected_ss)
    assert_allclose(model._evaluate_steady_state(), np.zeros(n), atol=1e-8)

    # Total error and gradient should be zero at the steady state as well
    error = model.f_ss_error(**params, **ss_dict)
    grad = model.f_ss_error_grad(**params, **ss_dict)
    hess = model.f_ss_error_hess(**params, **ss_dict)

    assert isinstance(error, float)
    assert isinstance(grad, np.ndarray)
    assert isinstance(hess, np.ndarray)

    assert grad.ndim == 1
    assert hess.ndim == 2

    assert_allclose(error, 0.0, atol=1e-8)
    assert_allclose(grad.ravel(), np.zeros((n,)), atol=1e-8)

    # Hessian should be PSD at the minimum (since it's a convex function)
    assert np.all(np.linalg.eigvals(hess) > -1e8)


@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
@pytest.mark.parametrize(
    "gcn_file",
    ["One_Block_Simple_1_w_Steady_State.gcn", "Open_RBC.gcn", "Full_New_Keynesian.gcn"],
)
def test_model_gradient(load_and_cache_model, backend, gcn_file):
    model = load_and_cache_model(gcn_file, backend)

    ss_result, success = model.steady_state()

    np.testing.assert_allclose(
        model.f_ss_error_grad(**ss_result, **model.parameters()),
        0.0,
        rtol=1e-12,
        atol=1e-12,
    )

    perturbed_point = {k: 0.8 for k, v in ss_result.items()}
    test_point = np.array(list(perturbed_point.values()))

    grad = model.f_ss_error_grad(**perturbed_point, **model.parameters())
    numeric_grad = nd.Gradient(lambda x: model.f_ss_error(*x, **model.parameters()))(
        test_point
    )

    np.testing.assert_allclose(grad, numeric_grad, rtol=1e-8, atol=1e-8)

    hess = model.f_ss_error_hess(**perturbed_point, **model.parameters())
    numeric_hess = nd.Hessian(lambda x: model.f_ss_error(*x, **model.parameters()))(
        test_point
    )

    np.testing.assert_allclose(hess, numeric_hess, rtol=1e-8, atol=1e-8)

    jac = model.f_ss_jac(**perturbed_point, **model.parameters())
    numeric_jac = nd.Jacobian(lambda x: model.f_ss_resid(*x, **model.parameters()))(
        test_point
    )

    np.testing.assert_allclose(jac, numeric_jac, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("how", ["root", "minimize"], ids=["root", "minimize"])
@pytest.mark.parametrize(
    "gcn_file",
    [
        "One_Block_Simple_1_w_Steady_State.gcn",
        "Open_RBC.gcn",
        pytest.param(
            "Full_New_Keynesian.gcn",
            marks=pytest.mark.skip("NK needs to be tuned to find SS without help"),
        ),
    ],
)
@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
def test_numerical_steady_state(
    load_and_cache_model, how: str, gcn_file: str, backend: BACKENDS
):
    model = load_and_cache_model(gcn_file, backend)
    analytic_res, success = model.steady_state()
    analytic_values = np.array([analytic_res[x.to_ss().name] for x in model.variables])

    # Overwrite the f_ss function with None to trigger numerical optimization
    # Save it so we can put it back later, or else the cached model won't have a steady state function anymore
    f_ss = model.f_ss
    model.f_ss = None

    x0 = np.full_like(analytic_values, 0.8)
    numeric_res, success = model.steady_state(
        how=how,
        verbose=False,
        optimizer_kwargs={
            "x0": x0,
            "tol": 1e-12,
            "method": "trust-constr" if how == "minimize" else "hybr",
            "options": {"maxiter": 10_000} if how == "minimize" else {},
        },
    )

    # Restore steady state function in the cached function
    model.f_ss = f_ss

    numeric_values = np.array([numeric_res[x.to_ss().name] for x in model.variables])
    errors = model.f_ss_resid(**numeric_res, **model.parameters())

    if how == "root":
        assert_allclose(analytic_values, numeric_values, atol=1e-2)
    elif how == "minimize":
        assert_allclose(errors, np.zeros_like(errors), atol=1e-2)


def test_numerical_steady_state_with_calibrated_params():
    file_path = "tests/Test GCNs/One_Block_Simple_2_without_Extra_Params.gcn"
    model = model_from_gcn(
        file_path, verbose=False, backend="numpy", mode="FAST_COMPILE"
    )
    res, success = model.steady_state(
        how="minimize",
        verbose=False,
        optimizer_kwargs={"method": "trust-constr", "options": {"maxiter": 100_000}},
        bounds={"alpha": (0.05, 0.7)},
    )
    res = res.to_string()
    assert_allclose(res["L_ss"] / res["K_ss"], 0.36)


@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
@pytest.mark.parametrize(
    "partial_file, analytic_file",
    [
        (
            "Two_Block_RBC_w_Partial_Steady_State.gcn",
            "Two_Block_RBC_w_Steady_State.gcn",
        ),
        ("Full_New_Keynesian_w_Partial_Steady_State.gcn", "Full_New_Keynesian.gcn"),
    ],
)
def test_partially_analytical_steady_state(
    load_and_cache_model, backend: BACKENDS, partial_file, analytic_file
):
    analytic_model = load_and_cache_model(analytic_file, backend)
    analytic_res, success = analytic_model.steady_state()
    analytic_values = np.array(list(analytic_res.values()))

    partial_model = load_and_cache_model(partial_file, backend)
    numeric_res, success = partial_model.steady_state(
        how="minimize",
        verbose=False,
        optimizer_kwargs={"method": "trust-ncg", "options": {"gtol": 1e-24}},
    )

    numeric_values = np.array(list(numeric_res.values()))

    errors = partial_model.f_ss_resid(
        **numeric_res.to_string(), **partial_model.parameters().to_string()
    )
    resid = partial_model.f_ss_resid(
        **numeric_res.to_string(), **partial_model.parameters().to_string()
    )

    ATOL = RTOL = 1e-1
    if partial_file == "Two_Block_RBC_w_Partial_Steady_State":
        assert_allclose(analytic_values, numeric_values, atol=ATOL, rtol=RTOL)

    assert_allclose(resid, 0, atol=ATOL, rtol=RTOL)
    assert_allclose(errors, np.zeros_like(errors), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "gcn_file, name",
    [
        ("One_Block_Simple_1_w_Steady_State.gcn", "one_block_ss"),
        ("Two_Block_RBC_w_Steady_State.gcn", "two_block_ss"),
        ("Full_New_Keynesian.gcn", "full_nk"),
    ],
    ids=["one_block_ss", "two_block_ss", "full_nk"],
)
@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
def test_linearize(load_and_cache_model, gcn_file, name, backend: BACKENDS):
    model = load_and_cache_model(gcn_file, backend)
    steady_state_dict, success = model.steady_state()
    outputs = model.linearize_model(
        loglin_negative_ss=True, steady_state_dict=steady_state_dict
    )

    for mat_name, out in zip(["A", "B", "C", "D"], outputs):
        expected_out = expected_linearization_result[gcn_file][mat_name]
        assert_allclose(out, expected_out, atol=1e-8, err_msg=f"{mat_name} failed")


def test_invalid_solver_raises():
    file_path = "tests/Test GCNs/One_Block_Simple_1_w_Steady_State.gcn"
    model = model_from_gcn(file_path, verbose=False)
    model.steady_state(verbose=False)

    with pytest.raises(NotImplementedError):
        model.solve_model(solver="invalid_solver")


def test_bad_failure_argument_raises():
    file_path = "tests/Test GCNs/pert_fails.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")

    with pytest.raises(ValueError):
        model.solve_model(solver="gensys", on_failure="raise", model_is_linear=True)


def test_gensys_fails_to_solve():
    file_path = "tests/Test GCNs/pert_fails.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")

    with pytest.raises(GensysFailedException):
        model.solve_model(solver="gensys", on_failure="error", verbose=False)


def test_outputs_after_gensys_failure(caplog):
    file_path = "tests/Test GCNs/pert_fails.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")
    model.solve_model(solver="gensys", on_failure="ignore", verbose=True)

    captured_message = caplog.messages[-1]
    assert captured_message == "Solution exists, but is not unique."

    P, Q, R, S = model.P, model.Q, model.R, model.S
    for X, name in zip([P, Q, R, S], ["P", "Q", "R", "S"]):
        assert X is None


def test_outputs_after_pert_success(caplog):
    file_path = "tests/Test GCNs/RBC_Linearized.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")
    model.solve_model(solver="gensys", verbose=True)

    result_messages = caplog.messages[-2:]
    expected_messages = [
        "Norm of deterministic part: 0.000000000",
        "Norm of stochastic part:    0.000000000",
    ]

    for message, expected_message in zip(result_messages, expected_messages):
        assert message == expected_message


def test_bad_argument_to_bk_condition_raises():
    file_path = "tests/Test GCNs/RBC_Linearized.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")
    model.solve_model(solver="gensys", verbose=False)

    with pytest.raises(ValueError, match='Unknown return type "invalid_argument"'):
        model.check_bk_condition(return_value="invalid_argument")


def test_check_bk_condition():
    file_path = "tests/Test GCNs/RBC_Linearized.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")
    model.solve_model(solver="gensys", verbose=False)

    bk_df = model.check_bk_condition(return_value="dataframe")
    assert isinstance(bk_df, pd.DataFrame)
    assert_allclose(
        bk_df["Modulus"].values,
        np.abs(bk_df["Real"].values + bk_df["Imaginary"].values * 1j),
    )

    bk_res = model.check_bk_condition(return_value="bool")
    assert bk_res


# def test_compute_stationary_covariance_warns_if_using_default(self):
#     file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn")
#     model = gEconModel(file_path, verbose=False)
#     model.steady_state(verbose=False)
#     model.solve_model(solver="gensys", verbose=False)
#
#     with self.assertWarns(UserWarning):
#         model.compute_stationary_covariance_matrix()
#
#
# def test_sample_priors_fails_without_priors(self):
#     file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn")
#     model = gEconModel(file_path, verbose=False)
#     model.steady_state(verbose=False)
#     model.solve_model(solver="gensys", verbose=False)
#
#     with self.assertRaises(ValueError):
#         model.sample_param_dict_from_prior()
#
#
# def test_missing_parameter_definition_raises(self):
#     GCN_file = """
#                 block HOUSEHOLD
#                 {
#                     definitions
#                     {
#                         u[] = log(C[]);
#                     };
#
#                     objective
#                     {
#                         U[] = u[] + beta * E[][U[1]];
#                     };
#
#                     controls
#                     {
#                         C[], K[], K[-1], Y[];
#                     };
#
#                     constraints
#                     {
#                         Y[] = K[-1] ^ alpha;
#                         Y[] = r[] * K[-1];
#                         K[] = (1 - delta) * K[-1];
#
#                     };
#
#                     calibration
#                     {
#                         K[ss] / Y[ss] = 0.33 -> alpha;
#                         delta = 0.035;
#                     };
#                 };
#                 """
#
#     with unittest.mock.patch(
#         "builtins.open",
#         new=unittest.mock.mock_open(read_data=GCN_file),
#         create=True,
#     ):
#         with self.assertRaises(ValueError) as error:
#             gEconModel(
#                 "",
#                 verbose=False,
#                 simplify_tryreduce=False,
#                 simplify_constants=False,
#             )
#         msg = str(error.exception)
#
#     self.assertEqual(
#         msg,
#         "The following parameters were found among model equations, but were not found among "
#         "defined defined or calibrated parameters: beta.\n Verify that these "
#         "parameters have been defined in a calibration block somewhere in your GCN file.",
#     )
#


#
# class ModelErrorTests(unittest.TestCase):
#     def setUp(self):
#         self.GCN_file = """
#             block HOUSEHOLD
#             {
#                 definitions
#                 {
#                     u[] = log(C[]);
#                 };
#
#                 objective
#                 {
#                     U[] = u[] + beta * E[][U[1]];
#                 };
#
#                 controls
#                 {
#                     C[], K[];
#                 };
#
#                 constraints
#                 {
#                     Y[] = K[-1] ^ alpha;
#                     C[] = r[] * K[-1];
#                     K[] = (1 - delta) * K[-1];
#                     X[] = Y[] + C[];
#                     Z[] = 3;
#                 };
#
#                 calibration
#                 {
#                     alpha = 0.33;
#                     beta = 0.99;
#                     delta = 0.035;
#                 };
#             };
#             """
#
#     def test_build_warns_if_model_not_defined(self):
#         expected_warnings = [
#             "Simplification via try_reduce was requested but not possible because the system is not well defined.",
#             "Removal of constant variables was requested but not possible because the system is not well defined.",
#             "The model does not appear correctly specified, there are 8 equations but "
#             "11 variables. It will not be possible to solve this model. Please check the "
#             "specification using available diagnostic tools, and check the GCN file for typos.",
#         ]
#
#         with unittest.mock.patch(
#             "builtins.open",
#             new=unittest.mock.mock_open(read_data=self.GCN_file),
#             create=True,
#         ):
#             with self.assertWarns(UserWarning) as warnings:
#                 model = gEconModel(
#                     "", verbose=False, simplify_tryreduce=True, simplify_constants=True
#                 )
#
#             for w in warnings.warnings:
#                 warning_msg = str(w.message)
#                 self.assertIn(warning_msg, expected_warnings)
#
#     def test_invalid_solver_raises(self):
#         file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_2.gcn")
#         model = gEconModel(file_path, verbose=False)
#         model.steady_state(verbose=False)
#
#         with self.assertRaises(NotImplementedError):
#             model.solve_model(solver="invalid_solver")
#
#     def test_bad_failure_argument_raises(self):
#         file_path = os.path.join(ROOT, "Test GCNs/pert_fails.gcn")
#         model = gEconModel(file_path, verbose=False)
#         model.steady_state(verbose=False, model_is_linear=True)
#
#         with self.assertRaises(ValueError):
#             model.solve_model(solver="gensys", on_failure="raise", model_is_linear=True)
#
#     def test_bad_argument_to_bk_condition_raises(self):
#         file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_2.gcn")
#         model = gEconModel(file_path, verbose=False)
#         model.steady_state(verbose=False)
#         model.solve_model(verbose=False)
#
#         with self.assertRaises(ValueError):
#             model.check_bk_condition(return_value="invalid_argument")
#
#     def test_gensys_fails_to_solve(self):
#         file_path = os.path.join(ROOT, "Test GCNs/pert_fails.gcn")
#         model = gEconModel(file_path, verbose=False)
#         model.steady_state(verbose=False, model_is_linear=True)
#
#         with self.assertRaises(GensysFailedException):
#             model.solve_model(
#                 solver="gensys", on_failure="error", model_is_linear=True, verbose=False
#             )
#
#     @mock.patch("builtins.print")
#     def test_outputs_after_gensys_failure(self, mock_print):
#         file_path = os.path.join(ROOT, "Test GCNs/pert_fails.gcn")
#         model = gEconModel(file_path, verbose=False)
#         model.steady_state(verbose=False, model_is_linear=True)
#         model.solve_model(solver="gensys", on_failure="ignore", model_is_linear=True, verbose=True)
#
#         gensys_message = mock_print.call_args.args[0]
#         self.assertEqual(gensys_message, "Solution exists, but is not unique.")
#
#         P, Q, R, S = model.P, model.Q, model.R, model.S
#         for X, name in zip([P, Q, R, S], ["P", "Q", "R", "S"]):
#             self.assertIsNone(X, msg=name)
#
#     @mock.patch("builtins.print")
#     def test_outputs_after_pert_success(self, mock_print):
#         file_path = os.path.join(ROOT, "Test GCNs/RBC_Linearized.gcn")
#         model = gEconModel(file_path, verbose=False)
#         model.steady_state(verbose=False, model_is_linear=True)
#         model.solve_model(solver="gensys", verbose=True, model_is_linear=True)
#
#         # TODO: Can i get more print calls without having to parse through call_args_list?
#         result_messages = mock_print.call_args.args[0]
#         self.assertEqual(result_messages, "Norm of stochastic part:    0.000000000")
#
#     def test_compute_stationary_covariance_warns_if_using_default(self):
#         file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn")
#         model = gEconModel(file_path, verbose=False)
#         model.steady_state(verbose=False)
#         model.solve_model(solver="gensys", verbose=False)
#
#         with self.assertWarns(UserWarning):
#             Sigma = model.compute_stationary_covariance_matrix()
#
#     def test_sample_priors_fails_without_priors(self):
#         file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn")
#         model = gEconModel(file_path, verbose=False)
#         model.steady_state(verbose=False)
#         model.solve_model(solver="gensys", verbose=False)
#
#         with self.assertRaises(ValueError):
#             model.sample_param_dict_from_prior()
#
#     def test_missing_parameter_definition_raises(self):
#         GCN_file = """
#                     block HOUSEHOLD
#                     {
#                         definitions
#                         {
#                             u[] = log(C[]);
#                         };
#
#                         objective
#                         {
#                             U[] = u[] + beta * E[][U[1]];
#                         };
#
#                         controls
#                         {
#                             C[], K[], K[-1], Y[];
#                         };
#
#                         constraints
#                         {
#                             Y[] = K[-1] ^ alpha;
#                             Y[] = r[] * K[-1];
#                             K[] = (1 - delta) * K[-1];
#
#                         };
#
#                         calibration
#                         {
#                             K[ss] / Y[ss] = 0.33 -> alpha;
#                             delta = 0.035;
#                         };
#                     };
#                     """
#
#         with unittest.mock.patch(
#             "builtins.open",
#             new=unittest.mock.mock_open(read_data=GCN_file),
#             create=True,
#         ):
#             with self.assertRaises(ValueError) as error:
#                 model = gEconModel(
#                     "",
#                     verbose=False,
#                     simplify_tryreduce=False,
#                     simplify_constants=False,
#                 )
#             msg = str(error.exception)
#
#         self.assertEqual(
#             msg,
#             "The following parameters were found among model equations, but were not found among "
#             "defined defined or calibrated parameters: beta.\n Verify that these "
#             "parameters have been defined in a calibration block somewhere in your GCN file.",
#         )
#
#
# class ModelClassTestsOne(unittest.TestCase):
#     def setUp(self):
#         file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_2.gcn")
#         self.model = gEconModel(file_path, verbose=False)
#
#     @unittest.mock.patch("builtins.print")
#     def test_build_report(self, mock_print):
#         self.model.build_report(reduced_vars=["A"], singletons=["B"], verbose=True)
#
#         expected_output = """
#             Model Building Complete.
#             Found:
#                 9 equations
#                 9 variables
#                 The following variables were eliminated at user request:
#                     A
#                 The following "variables" were defined as constants and have been substituted away:
#                     B
#                 1 stochastic shock
#                     0 / 1 has a defined prior.
#                 5 parameters
#                     0 / 5 has a defined prior.
#                 1 calibrating equation
#                 1 parameter to calibrate
#                 Model appears well defined and ready to proceed to solving."""
#         report = mock_print.call_args.args[0]
#
#         simple_output = re.sub("[\n\t]", " ", expected_output)
#         simple_output = re.sub(" +", " ", simple_output)
#
#         simple_report = re.sub("[\n\t]", " ", report)
#         simple_report = re.sub(" +", " ", simple_report)
#         self.assertEqual(simple_output.strip(), simple_report.strip())
#
#     def test_model_options(self):
#         self.assertEqual(self.model.options, {"output logfile": False, "output LaTeX": False})
#
#     def test_reduce_vars_saved(self):
#         self.assertEqual(
#             self.model.try_reduce_vars,
#             [TimeAwareSymbol("C", 0, **self.model.assumptions["C"])],
#         )
#
#     def test_model_file_loading(self):
#         block_names = ["HOUSEHOLD"]
#         result = [block_name for block_name in self.model.blocks.keys()]
#         self.assertEqual(block_names, result)
#
#         param_dict = {
#             "theta": 0.357,
#             "beta": 0.99,
#             "delta": 0.02,
#             "tau": 2,
#             "rho": 0.95,
#         }
#
#         self.assertEqual(
#             all([x in param_dict.keys() for x in self.model.free_param_dict.keys()]),
#             True,
#         )
#         self.assertEqual(
#             all([self.model.free_param_dict[x] == param_dict[x] for x in param_dict.keys()]),
#             True,
#         )
#         self.assertEqual(
#             self.model.params_to_calibrate,
#             [sp.Symbol("alpha", **self.model.assumptions["alpha"])],
#         )
#
#     def test_conflicting_assumptions_are_removed(self):
#         with self.assertWarns(UserWarning):
#             model = gEconModel(
#                 os.path.join(ROOT, "Test GCNs/conflicting_assumptions.gcn"),
#                 verbose=False,
#             )
#
#         self.assertTrue("real" not in model.assumptions["TC"].keys())
#         self.assertTrue("imaginary" in model.assumptions["TC"].keys())
#         self.assertTrue(model.assumptions["TC"]["imaginary"])
#
#     def test_solve_model_gensys(self):
#         self.setUp()
#         self.model.steady_state(verbose=False)
#         self.assertEqual(self.model.steady_state_solved, True)
#         self.model.solve_model(verbose=False, solver="gensys")
#         self.assertEqual(self.model.perturbation_solved, True)
#
#         # Values from R gEcon solution
#         P = np.array([[0.950, 0.0000], [0.2710273, 0.8916969]])
#
#         Q = np.array([[1.000], [0.2852917]])
#
#         # TODO: Bug? When the SS value is negative, the sign of the S and R matrix entries are flipped relative to
#         #   those of gEcon (row 4 -- Utility). This code flips the sign on my values to make the comparison.
#         #   Check Dynare.
#         R = np.array(
#             [
#                 [0.70641931, 0.162459910],
#                 [13.55135517, -4.415155354],
#                 [0.42838971, -0.152667442],
#                 [-0.06008706, -0.009473984],
#                 [1.36634369, -0.072720705],
#                 [-0.80973441, -0.273514035],
#                 [-0.80973441, -0.273514035],
#             ]
#         )
#
#         S = np.array(
#             [
#                 [0.74359928],
#                 [14.26458439],
#                 [0.45093654],
#                 [-0.06324954],
#                 [1.43825652],
#                 [-0.85235201],
#                 [-0.85235201],
#             ]
#         )
#
#         ss_df = pd.Series(string_keys_to_sympy(self.model.steady_state_dict))
#         ss_df.index = list(map(lambda x: x.exit_ss().name, ss_df.index))
#         # ss_df = ss_df.reindex(self.model.S.index)
#         # neg_ss_mask = ss_df < 0
#
#         A, _, _, _ = self.model.build_perturbation_matrices(
#             np.fromiter(
#                 (self.model.free_param_dict | self.model.calib_param_dict).values(),
#                 dtype="float",
#             ),
#             np.fromiter(self.model.steady_state_dict.values(), dtype="float"),
#         )
#
#         (
#             _,
#             variables,
#             _,
#         ) = self.model.perturbation_solver.make_all_variable_time_combinations()
#
#         gEcon_matrices = self.model.perturbation_solver.statespace_to_gEcon_representation(
#             A, self.model.T.values, self.model.R.values, variables, 1e-7
#         )
#         model_P, model_Q, model_R, model_S, *_ = gEcon_matrices
#
#         assert_allclose(model_P, P, equal_nan=True, err_msg="P", rtol=1e-5)
#         assert_allclose(model_Q, Q, equal_nan=True, err_msg="Q", rtol=1e-5)
#         assert_allclose(model_R, R, equal_nan=True, err_msg="R", rtol=1e-5)
#         assert_allclose(model_S, S, equal_nan=True, err_msg="S", rtol=1e-5)
#
#     def test_solve_model_cycle_reduction(self):
#         self.setUp()
#         self.model.steady_state(verbose=True)
#         self.assertEqual(self.model.steady_state_solved, True)
#         self.model.solve_model(verbose=True, solver="cycle_reduction")
#         self.assertEqual(self.model.perturbation_solved, True)
#
#         # Values from R gEcon solution
#         P = np.array([[0.950, 0.0000], [0.2710273, 0.8916969]])
#
#         Q = np.array([[1.000], [0.2852917]])
#
#         # TODO: Check dynare outputs for sign flip
#         R = np.array(
#             [
#                 [0.70641931, 0.162459910],
#                 [13.55135517, -4.415155354],
#                 [0.42838971, -0.152667442],
#                 [-0.06008706, -0.009473984],
#                 [1.36634369, -0.072720705],
#                 [-0.80973441, -0.273514035],
#                 [-0.80973441, -0.273514035],
#             ]
#         )
#
#         S = np.array(
#             [
#                 [0.74359928],
#                 [14.26458439],
#                 [0.45093654],
#                 [-0.06324954],
#                 [1.43825652],
#                 [-0.85235201],
#                 [-0.85235201],
#             ]
#         )
#
#         A, _, _, _ = self.model.build_perturbation_matrices(
#             np.fromiter(
#                 (self.model.free_param_dict | self.model.calib_param_dict).values(),
#                 dtype="float",
#             ),
#             np.fromiter(self.model.steady_state_dict.values(), dtype="float"),
#         )
#
#         (
#             _,
#             variables,
#             _,
#         ) = self.model.perturbation_solver.make_all_variable_time_combinations()
#
#         gEcon_matrices = self.model.perturbation_solver.statespace_to_gEcon_representation(
#             A, self.model.T.values, self.model.R.values, variables, 1e-7
#         )
#         model_P, model_Q, model_R, model_S, *_ = gEcon_matrices
#
#         self.assertEqual(np.allclose(model_P, P), True, msg="P")
#         self.assertEqual(np.allclose(model_Q, Q), True, msg="Q")
#         self.assertEqual(np.allclose(model_R, R), True, msg="R")
#         self.assertEqual(np.allclose(model_S, S), True, msg="S")
#
#     def test_solvers_agree(self):
#         self.setUp()
#         self.model.steady_state(verbose=False)
#         self.model.solve_model(solver="gensys", verbose=False)
#         Tg, Rg = self.model.T, self.model.R
#
#         self.setUp()
#         self.model.steady_state(verbose=False)
#         self.model.solve_model(solver="cycle_reduction", verbose=False)
#         Tc, Rc = self.model.T, self.model.R
#
#         assert_allclose(
#             Tg.round(5).values,
#             Tc.round(5).values,
#             rtol=1e-5,
#             equal_nan=True,
#             err_msg="T",
#         )
#         assert_allclose(
#             Rg.round(5).values,
#             Rc.round(5).values,
#             rtol=1e-5,
#             equal_nan=True,
#             err_msg="R",
#         )
#
#     def test_blanchard_kahn_conditions(self):
#         self.model.steady_state(verbose=False)
#         self.model.solve_model(verbose=False)
#         bk_cond = self.model.check_bk_condition(return_value="bool", verbose=True)
#         self.assertTrue(bk_cond)
#
#         bk_df = self.model.check_bk_condition(return_value="df")
#         self.assertTrue(isinstance(bk_df, pd.DataFrame))
#
#     def test_compute_autocorrelation_matrix(self):
#         self.model.steady_state(verbose=False)
#         self.model.solve_model(verbose=False)
#
#         n_lags = 10
#         acorr_df = self.model.compute_autocorrelation_matrix(
#             shock_dict={"epsilon_A": 0.01}, n_lags=n_lags
#         )
#
#         self.assertTrue(isinstance(acorr_df, pd.DataFrame))
#         self.assertEqual(acorr_df.shape[0], self.model.n_variables)
#         self.assertEqual(acorr_df.shape[1], n_lags)
#
#     def test_compute_stationary_covariance(self):
#         self.model.steady_state(verbose=False)
#         self.model.solve_model(verbose=False)
#
#         Sigma = self.model.compute_stationary_covariance_matrix(shock_dict={"epsilon_A": 0.01})
#         self.assertTrue(isinstance(Sigma, pd.DataFrame))
#         self.assertTrue(all([x == self.model.n_variables for x in Sigma.shape]))
#
#
# class ModelClassTestsTwo(unittest.TestCase):
#     def setUp(self):
#         file_path = os.path.join(ROOT, "Test GCNs/Two_Block_RBC_1.gcn")
#         self.model = gEconModel(file_path, verbose=False)
#
#     def test_model_options(self):
#         self.assertEqual(
#             self.model.options,
#             {
#                 "output logfile": True,
#                 "output LaTeX": True,
#                 "output LaTeX landscape": True,
#             },
#         )
#
#     def test_reduce_vars_saved(self):
#         self.assertEqual(self.model.try_reduce_vars, None)
#
#     def test_model_file_loading(self):
#         block_names = ["HOUSEHOLD", "FIRM"]
#         result = [block_name for block_name in self.model.blocks.keys()]
#         self.assertEqual(result, block_names)
#
#         param_dict = {
#             "beta": 0.985,
#             "delta": 0.025,
#             "sigma_C": 2,
#             "sigma_L": 1.5,
#             "alpha": 0.35,
#             "rho_A": 0.95,
#         }
#
#         self.assertEqual(
#             all([self.model.free_param_dict[x] == param_dict[x] for x in param_dict.keys()]),
#             True,
#         )
#         self.assertEqual(self.model.params_to_calibrate, [])
#
#     def test_solve_model_gensys(self):
#         self.model.steady_state(verbose=False)
#         self.assertEqual(self.model.steady_state_solved, True)
#         self.model.solve_model(verbose=False, solver="gensys")
#         self.assertEqual(self.model.perturbation_solved, True)
#
#         P = np.array([[0.95000000, 0.0000000], [0.08887552, 0.9614003]])
#
#         Q = np.array([[1.00000000], [0.09355318]])
#
#         # TODO: Investigate sign flip on row 5, 6 (TC, U)
#         R = np.array(
#             [
#                 [0.3437521, 0.3981261],
#                 [3.5550207, -0.5439888],
#                 [0.1418896, -0.2412174],
#                 [1.0422283, 0.1932087],
#                 [-0.2127497, -0.1270917],
#                 [1.0422282, 0.1932087],
#                 [-0.6875042, -0.7962522],
#                 [-0.6875042, -0.7962522],
#                 [1.0422284, -0.8067914],
#                 [0.9003386, 0.4344261],
#             ]
#         )
#
#         S = np.array(
#             [
#                 [0.3618443],
#                 [3.7421271],
#                 [0.1493575],
#                 [1.0970824],
#                 [-0.2239471],
#                 [1.0970823],
#                 [-0.7236886],
#                 [-0.7236886],
#                 [1.0970825],
#                 [0.9477249],
#             ]
#         )
#
#         A, _, _, _ = self.model.build_perturbation_matrices(
#             np.fromiter(
#                 (self.model.free_param_dict | self.model.calib_param_dict).values(),
#                 dtype="float",
#             ),
#             np.fromiter(self.model.steady_state_dict.values(), dtype="float"),
#         )
#
#         (
#             _,
#             variables,
#             _,
#         ) = self.model.perturbation_solver.make_all_variable_time_combinations()
#
#         gEcon_matrices = self.model.perturbation_solver.statespace_to_gEcon_representation(
#             A, self.model.T.values, self.model.R.values, variables, 1e-7
#         )
#         model_P, model_Q, model_R, model_S, *_ = gEcon_matrices
#
#         assert_allclose(model_P, P, equal_nan=True, err_msg="P", rtol=1e-5)
#         assert_allclose(model_Q, Q, equal_nan=True, err_msg="Q", rtol=1e-5)
#         assert_allclose(model_R, R, equal_nan=True, err_msg="R", rtol=1e-5)
#         assert_allclose(model_S, S, equal_nan=True, err_msg="S", rtol=1e-5)
#
#     def test_solve_model_cycle_reduction(self):
#         self.model.steady_state(verbose=False)
#         self.assertEqual(self.model.steady_state_solved, True)
#         self.model.solve_model(verbose=False, solver="cycle_reduction")
#
#         P = np.array([[0.95000000, 0.0000000], [0.08887552, 0.9614003]])
#
#         Q = np.array([[1.00000000], [0.09355318]])
#
#         # TODO: Investigate sign flip on row 5, 6 (TC, U)
#         R = np.array(
#             [
#                 [0.3437521, 0.3981261],
#                 [3.5550207, -0.5439888],
#                 [0.1418896, -0.2412174],
#                 [1.0422283, 0.1932087],
#                 [-0.2127497, -0.1270917],
#                 [1.0422282, 0.1932087],
#                 [-0.6875042, -0.7962522],
#                 [-0.6875042, -0.7962522],
#                 [1.0422284, -0.8067914],
#                 [0.9003386, 0.4344261],
#             ]
#         )
#
#         S = np.array(
#             [
#                 [0.3618443],
#                 [3.7421271],
#                 [0.1493575],
#                 [1.0970824],
#                 [-0.2239471],
#                 [1.0970823],
#                 [-0.7236886],
#                 [-0.7236886],
#                 [1.0970825],
#                 [0.9477249],
#             ]
#         )
#
#         A, _, _, _ = self.model.build_perturbation_matrices(
#             np.fromiter(
#                 (self.model.free_param_dict | self.model.calib_param_dict).values(),
#                 dtype="float",
#             ),
#             np.fromiter(self.model.steady_state_dict.values(), dtype="float"),
#         )
#
#         (
#             _,
#             variables,
#             _,
#         ) = self.model.perturbation_solver.make_all_variable_time_combinations()
#
#         gEcon_matrices = self.model.perturbation_solver.statespace_to_gEcon_representation(
#             A, self.model.T.values, self.model.R.values, variables, 1e-7
#         )
#         model_P, model_Q, model_R, model_S, *_ = gEcon_matrices
#
#         assert_allclose(model_P, P, equal_nan=True, err_msg="P", rtol=1e-5)
#         assert_allclose(model_Q, Q, equal_nan=True, err_msg="Q", rtol=1e-5)
#         assert_allclose(model_R, R, equal_nan=True, err_msg="R", rtol=1e-5)
#         assert_allclose(model_S, S, equal_nan=True, err_msg="S", rtol=1e-5)
#
#     def test_solvers_agree(self):
#         self.setUp()
#         self.model.steady_state(verbose=False)
#         self.model.solve_model(solver="gensys", verbose=False)
#         Tg, Rg = self.model.T, self.model.R
#
#         self.setUp()
#         self.model.steady_state(verbose=False)
#         self.model.solve_model(solver="cycle_reduction", verbose=False)
#         Tc, Rc = self.model.T, self.model.R
#
#         assert_allclose(
#             Tg.round(5).values,
#             Tc.round(5).values,
#             rtol=1e-5,
#             equal_nan=True,
#             err_msg="T",
#         )
#         assert_allclose(
#             Rg.round(5).values,
#             Rc.round(5).values,
#             rtol=1e-5,
#             equal_nan=True,
#             err_msg="R",
#         )
#
#
# class ModelClassTestsThree(unittest.TestCase):
#     def setUp(self):
#         file_path = os.path.join(ROOT, "Test GCNs/Full_New_Keynesian.gcn")
#         self.model = gEconModel(
#             file_path, verbose=False, simplify_constants=False, simplify_tryreduce=False
#         )
#
#     def test_model_options(self):
#         self.assertEqual(
#             self.model.options,
#             {
#                 "output logfile": True,
#                 "output LaTeX": True,
#                 "output LaTeX landscape": True,
#             },
#         )
#
#     def test_reduce_vars_saved(self):
#         self.assertEqual(
#             self.model.try_reduce_vars,
#             [
#                 "Div[]",
#                 "TC[]",
#                 # TimeAwareSymbol("Div", 0, **self.model.assumptions["DIV"]),
#                 # TimeAwareSymbol("TC", 0, **self.model.assumptions["TC"]),
#             ],
#         )
#
#     def test_model_file_loading(self):
#         block_names = [
#             "HOUSEHOLD",
#             "WAGE_SETTING",
#             "WAGE_EVOLUTION",
#             "PREFERENCE_SHOCKS",
#             "FIRM",
#             "TECHNOLOGY_SHOCKS",
#             "FIRM_PRICE_SETTING_PROBLEM",
#             "PRICE_EVOLUTION",
#             "MONETARY_POLICY",
#             "EQUILIBRIUM",
#         ]
#
#         result = [block_name for block_name in self.model.blocks.keys()]
#         self.assertEqual(result, block_names)
#
#         (
#             rho_technology,
#             gamma_R,
#             gamma_pi,
#             gamma_Y,
#             phi_pi_obj,
#             phi_pi,
#             rho_pi_dot,
#         ) = sp.symbols(
#             [
#                 "rho_technology",
#                 "gamma_R",
#                 "gamma_pi",
#                 "gamma_Y",
#                 "phi_pi_obj",
#                 "phi_pi",
#                 "rho_pi_dot",
#             ],
#             **DEFAULT_ASSUMPTIONS,
#         )
#
#         param_dict = {
#             "delta": 0.025,
#             "beta": 0.99,
#             "sigma_C": 2,
#             "sigma_L": 1.5,
#             "gamma_I": 10,
#             "phi_H": 0.5,
#             "psi_w": 0.782,
#             "eta_w": 0.75,
#             "alpha": 0.35,
#             "rho_technology": 0.95,
#             "rho_preference": 0.95,
#             "psi_p": 0.6,
#             "eta_p": 0.75,
#             "gamma_R": 0.9,
#             "gamma_pi": 1.5,
#             "gamma_Y": 0.05,
#             "rho_pi_dot": 0.924,
#         }
#
#         self.assertEqual(
#             all([x in param_dict.keys() for x in self.model.free_param_dict.keys()]),
#             True,
#         )
#         self.assertEqual(
#             all([self.model.free_param_dict[x] == param_dict[x] for x in param_dict.keys()]),
#             True,
#         )
#         self.assertEqual(self.model.params_to_calibrate, [phi_pi, phi_pi_obj])
#
#     def test_solvers_agree(self):
#         self.setUp()
#         self.model.steady_state(verbose=False)
#         self.model.solve_model(solver="gensys", verbose=False)
#         Tg, Rg = self.model.T, self.model.R
#
#         self.setUp()
#         self.model.steady_state(verbose=False)
#         self.model.solve_model(solver="cycle_reduction", verbose=False)
#         Tc, Rc = self.model.T, self.model.R
#
#         assert_allclose(Tg.values, Tc.values, rtol=1e-5, atol=1e-5, equal_nan=True, err_msg="T")
#         assert_allclose(Rg.values, Rc.values, rtol=1e-5, atol=1e-5, equal_nan=True, err_msg="R")
#
#     # def test_solve_model(self):
#     #     self.model.steady_state(verbose=False)
#
#     #     self.model.solve_model(verbose=False, solver='gensys')
#     #
#     #     P = np.array([[0.92400000, 0.00000000, 0.000000000, 0.000000000, 0.000000000, 0.0000000000, 0.0000000000,
#     #                    0.000000000, 0.00000000, 0.0000000000],
#     #                   [0.04464553, 0.77386407, 0.008429303, -0.035640523, 0.019260369, -0.0061647545, 0.0064098938,
#     #                    0.003811426, -0.01635691, -0.0042992448],
#     #                   [0.00000000, 0.00000000, 0.950000000, 0.000000000, 0.000000000, 0.0000000000, 0.0000000000,
#     #                    0.000000000, 0.00000000, 0.0000000000],
#     #                   [0.00000000, 0.00000000, 0.000000000, 0.950000000, 0.000000000, 0.0000000000, 0.0000000000,
#     #                    0.000000000, 0.00000000, 0.0000000000],
#     #                   [0.11400712, -0.23033661, 0.017018503, 0.246571939, 0.714089188, 0.0015115630, -0.0025199985,
#     #                    0.003439315, 0.09953510, 0.0012796478],
#     #                   [0.00000000, 0.00000000, 0.000000000, 0.000000000, 0.000000000, 0.0000000000, 0.0000000000,
#     #                    0.000000000, 0.00000000, 0.0000000000],
#     #                   [0.56944713, -1.31534877, 0.116205871, 0.279528217, -0.069058930, 0.0055509980, 0.4892113664,
#     #                    0.001268753, 0.13710342, 0.0073074932],
#     #                   [0.77344786, -1.65448037, -0.084222852, 0.373554371, -0.110359402, 0.0067463467, -0.0129713461,
#     #                    0.893824526, -0.07071734, 0.0091915576],
#     #                   [0.01933620, -0.04136201, -0.002105571, 0.009338859, -0.002758985, 0.0001686587, -0.0003242837,
#     #                    0.022345613, 0.97323207, 0.0002297889],
#     #                   [0.60123052, -1.36818560, 0.084979004, 0.294177526, -0.075493558, -0.5547430352, 0.4109711206,
#     #                    0.140329261, 0.10472487, 0.0076010311]])
#     #
#     #     Q = np.array([[0.000000000, 0.000000000, 0.00000000, 1.00000000],
#     #                   [0.008872950, -0.037516340, 0.85984896, 0.04831767],
#     #                   [1.000000000, 0.000000000, 0.00000000, 0.00000000],
#     #                   [0.000000000, 1.000000000, 0.00000000, 0.00000000],
#     #                   [0.017914213, 0.259549409, -0.25592956, 0.12338433],
#     #                   [0.000000000, 0.000000000, 0.00000000, 0.00000000],
#     #                   [0.122321970, 0.294240229, -1.46149864, 0.61628477],
#     #                   [-0.088655634, 0.393215127, -1.83831153, 0.83706479],
#     #                   [-0.002216391, 0.009830378, -0.04595779, 0.02092662],
#     #                   [0.089451584, 0.309660553, -1.52020622, 0.65068238]])
#     #
#     #     R = np.array([[-2.70120790, 6.4759672, 0.45684368, -1.0523862, 0.25304694, -0.028589270, 0.043922008,
#     #                    -0.010211851, -0.50854833, -0.0359775957],
#     #                   [0.43774664, -0.9670519, 0.06277643, -1.0565632, 0.67343881, -0.297196226, 0.218772144,
#     #                    0.079001225, -0.38253612, 0.0053725107],
#     #                   [0.58559582, -0.7953000, 0.05336272, -0.2474094, 0.13091891, -0.022606929, 0.029033588,
#     #                    0.020731865, -0.11253692, 0.0044183336],
#     #                   [1.75678747, -2.3859001, 0.16008816, -0.7422282, 0.39275674, -0.067820786, 0.087100765,
#     #                    0.062195594, -0.33761076, 0.0132550007],
#     #                   [-0.34114299, 0.5424464, 0.48057739, -0.7361740, 0.12517618, -0.002047156, 0.028009363,
#     #                    -0.063210365, -0.75978505, -0.0030135913],
#     #                   [1.03897717, -2.3352376, 0.14775544, -0.7623857, 0.59794526, -0.851939276, 0.629743275,
#     #                    0.219330490, -1.27781127, 0.0129735420],
#     #                   [2.21281597, -3.3072465, 0.22816217, 0.2440595, 0.24911350, -0.061774534, 0.077020771,
#     #                    0.075952852, 0.06052965, 0.0183735919],
#     #                   [0.92497003, -2.1049009, 0.13073693, -1.0089577, -0.11614394, -0.853450826, 0.632263264,
#     #                    0.215891172, -0.37734635, 0.0116938940],
#     #                   [-1.86247082, 3.7798186, 0.45779728, -1.0016774, 0.69227986, -0.140940083, 0.177078457,
#     #                    0.079706624, -0.54739326, -0.0209989924],
#     #                   [2.76788546, -2.2659060, 0.88668501, -1.6781507, 0.64733010, -0.213012025, 0.289374259,
#     #                    0.186334186, -0.95855542, 0.0125883667],
#     #                   [-1.86247082, 3.7798186, 0.45779728, -1.0016774, 0.69227986, -0.140940083, 0.177078457,
#     #                    0.079706624, -0.54739326, -0.0209989924],
#     #                   [2.76788546, -2.2659060, 0.88668501, -1.6781507, 0.64733010, -0.213012025, 0.289374259,
#     #                    0.186334186, -0.95855542, 0.0125883667],
#     #                   [0.07745758, -0.1102706, -0.16967797, 0.1782883, -0.01302221, 0.002553406, -0.004433502,
#     #                    0.007300968, 0.07817343, 0.0006126142]])
#     #
#     #     S = np.array([[0.48088808, -1.1077749, 7.1955191, -2.92338518],
#     #                   [0.06608045, -1.1121718, -1.0745021, 0.47375177],
#     #                   [0.05617128, -0.2604309, -0.8836667, 0.63376171],
#     #                   [0.16851385, -0.7812928, -2.6510001, 1.90128514],
#     #                   [0.50587094, -0.7749200, 0.6027183, -0.36920237],
#     #                   [0.15553204, -0.8025113, -2.5947084, 1.12443417],
#     #                   [0.24017070, 0.2569048, -3.6747184, 2.39482247],
#     #                   [0.13761782, -1.0620607, -2.3387788, 1.00104982],
#     #                   [0.48189187, -1.0543973, 4.1997985, -2.01566106],
#     #                   [0.93335265, -1.7664744, -2.5176733, 2.99554704],
#     #                   [0.48189187, -1.0543973, 4.1997985, -2.01566106],
#     #                   [0.93335265, -1.7664744, -2.5176733, 2.99554704],
#     #                   [-0.17860839, 0.1876719, -0.1225228, 0.08382855]])
#     #
#     #     index_10 = ['pi_obj', 'r_G', 'shock_preference', 'shock_technology', 'w', 'B', 'C', 'I', 'K', 'Y']
#     #     cols_10 = ['pi_obj', 'r_G', 'shock_preference', 'shock_technology', 'w', 'B', 'C',  'I', 'K', 'Y']
#     #
#     #
#     #
#     #     index_11 = ['lambda_t', 'q_t', 'r_t', 'w_t', 'C_t', 'I_t', 'L_t', 'P_t', 'TC_t', 'U_t', 'Y_t']
#     #     ss_df = pd.Series(self.model.steady_state_dict)
#     #     ss_df.index = list(map(lambda x: x.exit_ss().name, ss_df.index))
#     #     ss_df = ss_df.reindex(self.model.S.index)
#     #     neg_ss_mask = ss_df < 0
#     #
#     #     for answer, result in zip([P, Q, R, S], [self.model.P, self.model.Q, self.model.R, self.model.S]):
#     #         if result.shape[0] == 11:
#     #             result = result.loc[index_11, :]
#     #             result.loc[neg_ss_mask, :] = result.loc[neg_ss_mask, :] * -1
#     #         self.assertEqual(np.allclose(answer, result.values), True)
#
#
# class TestLinearModel(unittest.TestCase):
#     def setUp(self):
#         file_path = os.path.join(ROOT, "Test GCNs/RBC_Linearized.gcn")
#         self.model = gEconModel(file_path, verbose=False)
#
#     def test_deterministics_are_extracted(self):
#         self.assertEqual(len(self.model.deterministic_params), 7)
#
#     def test_steady_state(self):
#         self.model.steady_state(model_is_linear=True, verbose=False)
#         self.assertTrue(self.model.steady_state_solved)
#         self.assertTrue(
#             np.allclose(
#                 np.array(list(self.model.steady_state_dict.values())),
#                 np.array([0, 0, 0, 0, 0, 0, 0, 0]),
#             )
#         )
#
#     def test_perturbation_solver(self):
#         self.model.steady_state(verbose=False, model_is_linear=True)
#         self.model.solve_model(verbose=False, model_is_linear=True)
#         self.assertTrue(self.model.perturbation_solved)
#
#         T_dynare = np.array(
#             [
#                 [0.95, 0.0],
#                 [0.34375208, 0.39812608],
#                 [3.55502044, -0.54398862],
#                 [0.08887551, 0.96140028],
#                 [0.14188965, -0.24121738],
#                 [1.04222827, -0.8067913],
#                 [0.90033862, 0.43442608],
#                 [1.04222827, 0.1932087],
#             ]
#         )
#
#         R_dynare = np.array(
#             [1.0, 0.361844, 3.742127, 0.093553, 0.149358, 1.097082, 0.947725, 1.097082]
#         )
#
#         assert_allclose(
#             self.model.T[["A", "K"]].values, T_dynare, rtol=1e-5, atol=1e-5, err_msg="T"
#         )
#         assert_allclose(
#             self.model.R.values,
#             R_dynare.reshape(-1, 1),
#             rtol=1e-5,
#             atol=1e-5,
#             err_msg="R",
#         )
#
#     def test_solvers_agree(self):
#         self.setUp()
#         self.model.steady_state(verbose=False, model_is_linear=True)
#         self.model.solve_model(solver="gensys", verbose=False, model_is_linear=True)
#         Tg, Rg = self.model.T, self.model.R
#
#         self.setUp()
#         self.model.steady_state(verbose=False, model_is_linear=True)
#         self.model.solve_model(solver="cycle_reduction", verbose=False, model_is_linear=True)
#         Tc, Rc = self.model.T, self.model.R
#
#         assert_allclose(Tg.values, Tc.values, rtol=1e-5, atol=1e-5, equal_nan=True, err_msg="T")
#         assert_allclose(Rg.values, Rc.values, rtol=1e-5, atol=1e-5, equal_nan=True, err_msg="R")
#
#
# class TestModelSimulationTools(unittest.TestCase):
#     def setUp(self):
#         file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_1_w_Distributions.gcn")
#         self.model = gEconModel(file_path, verbose=False)
#         self.model.steady_state(verbose=False)
#         self.model.solve_model(verbose=False)
#
#     def test_sample_param_dicts(self):
#         param_dict, shock_dict, obs_dict = self.model.sample_param_dict_from_prior(n_samples=100)
#
#         self.assertTrue(all([x in self.model.free_param_dict for x in param_dict.to_string()]))
#         self.assertTrue(len(param_dict) == 3)
#
#         self.assertTrue(all([x.name in shock_dict for x in self.model.shocks]))
#         self.assertTrue(len(shock_dict) == 1)
#
#         self.assertTrue(len(obs_dict) == 0)
#
#     def test_irf(self):
#         simulation_length = 40
#         irf = self.model.impulse_response_function(
#             simulation_length=simulation_length, shock_size=0.1
#         )
#
#         self.assertTrue(isinstance(irf, pd.DataFrame))
#         self.assertTrue(irf.shape[0] == self.model.n_variables)
#         self.assertTrue(irf.shape[1] == self.model.n_shocks * simulation_length)
#
#     def test_simulate_warns_on_defaults(self):
#         simulation_length = 40
#         n_simulations = 1
#
#         # Overwrite the priors to get the warning
#         self.model.hyper_priors = SymbolDictionary()
#         self.model.shock_priors = SymbolDictionary()
#         with self.assertWarns(UserWarning):
#             data = self.model.simulate(
#                 simulation_length=simulation_length, n_simulations=n_simulations
#             )
#
#     def test_simulate_from_covariance_matrix(self):
#         simulation_length = 40
#         n_simulations = 1
#         Q = np.array([[0.01]])
#         data = self.model.simulate(
#             simulation_length=simulation_length,
#             n_simulations=n_simulations,
#             shock_cov_matrix=Q,
#         )
#
#         self.assertTrue(isinstance(data, pd.DataFrame))
#         self.assertTrue(data.shape[0] == self.model.n_variables)
#         self.assertTrue(data.shape[1] == simulation_length * n_simulations)
#
#     def test_simulate_from_shock_dict(self):
#         simulation_length = 40
#         n_simulations = 1
#         shock_dict = {"epsilon_A": 0.1}
#         data = self.model.simulate(
#             simulation_length=simulation_length,
#             n_simulations=n_simulations,
#             shock_dict=shock_dict,
#         )
#
#         self.assertTrue(isinstance(data, pd.DataFrame))
#         self.assertTrue(data.shape[0] == self.model.n_variables)
#         self.assertTrue(data.shape[1] == simulation_length * n_simulations)
#
#     def test_fit_model_and_sample_posterior_trajectories(self):
#         T = 100
#         n_simulations = 1
#
#         # Draw from shock prior
#         data = self.model.simulate(simulation_length=T, n_simulations=n_simulations)
#
#         # Only Y is observed
#         data = data.droplevel(axis=1, level=1).T[["C"]]
#
#         idata = self.model.fit(
#             data,
#             filter_type="univariate",
#             draws=36,
#             n_walkers=36,
#             return_inferencedata=True,
#             burn_in=0,
#             verbose=False,
#             compute_sampler_stats=False,
#         )
#
#         self.assertIsNotNone(idata)
#
#         # Check posterior sampling. It should be its own test, but I want to minimize expensive model fitting calls
#         posterior = az.extract(idata, "posterior")
#         conditional_posterior = simulate_trajectories_from_posterior(
#             self.model, posterior, n_samples=10, n_simulations=10, simulation_length=10
#         )
#
#         self.assertIsNotNone(conditional_posterior)
#
#     def test_fit_model_raises_on_stochastic_singularity(self):
#         T = 100
#         n_simulations = 1
#
#         # Draw from shock prior
#         data = self.model.simulate(simulation_length=T, n_simulations=n_simulations)
#
#         # Only Y is observed
#         data = data.droplevel(axis=1, level=1).T[["C", "K"]]
#
#         with self.assertRaises(ValueError):
#             idata = self.model.fit(
#                 data,
#                 filter_type="univariate",
#                 draws=36,
#                 n_walkers=36,
#                 return_inferencedata=True,
#                 burn_in=0,
#                 verbose=False,
#                 compute_sampler_stats=False,
#             )
#
#
# if __name__ == "__main__":
#     unittest.main()
