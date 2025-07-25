import re
from importlib.util import find_spec

import numdifftools as nd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from gEconpy.exceptions import GensysFailedException
from gEconpy.model.build import model_from_gcn
from gEconpy.model.compile import BACKENDS
from gEconpy.model.model import (
    autocorrelation_matrix,
    build_Q_matrix,
    impulse_response_function,
    matrix_to_dataframe,
    scipy_wrapper,
    simulate,
    stationary_covariance_matrix,
    summarize_perturbation_solution,
)
from gEconpy.model.perturbation import (
    check_bk_condition,
)
from tests._resources.expected_matrices import expected_linearization_result
from tests._resources.load_dynare import load_dynare_outputs
from tests._resources.cache_compiled_models import load_and_cache_model

JAX_INSTALLED = find_spec("jax") is not None


@pytest.mark.parametrize(
    "gcn_path, name",
    [
        ("one_block_1_dist.gcn", "one_block_prior"),
        ("one_block_1_ss.gcn", "one_block_ss"),
        ("full_nk.gcn", "full_nk"),
    ],
    ids=["one_block_prior", "one_block_ss", "full_nk"],
)
@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
def test_model_parameters(gcn_path: str, name: str, backend: BACKENDS):
    model = load_and_cache_model(gcn_path, backend, use_jax=JAX_INSTALLED)

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
def test_deterministic_model_parameters(backend: BACKENDS):
    model = load_and_cache_model("one_block_2.gcn", backend, use_jax=JAX_INSTALLED)
    params = model.parameters()

    # Test numeric expression in calibration block
    assert_allclose(params["beta"], 1 / 1.01)

    # Test deterministic relationship
    params = model.parameters(theta=0.9)
    assert params["theta"] == 0.9
    assert_allclose(params["zeta"], -np.log(0.9))


@pytest.mark.parametrize(
    "gcn_path",
    ["one_block_1_ss.gcn", "open_rbc.gcn", "full_nk.gcn"],
    ids=["one_block_prior", "one_block_ss", "full_nk"],
)
def test_all_backends_agree_on_parameters(gcn_path):
    models = [
        load_and_cache_model(gcn_path, backend, use_jax=JAX_INSTALLED)
        for backend in ["numpy", "numba", "pytensor"]
    ]
    params = [np.r_[list(model.parameters().values())] for model in models]

    for i in range(3):
        for j in range(i):
            assert_allclose(params[i], params[j])


@pytest.mark.parametrize(
    "gcn_path",
    ["one_block_1_ss.gcn", "open_rbc.gcn", "full_nk.gcn"],
    ids=["one_block_prior", "one_block_ss", "full_nk"],
)
@pytest.mark.parametrize(
    "func",
    ["f_ss_error_grad", "f_ss_error_hess", "f_ss_jac"],
    ids=["grad", "hess", "jac"],
)
def test_all_backends_agree_on_functions(gcn_path, func):
    backends = ["numpy", "numba", "pytensor"]
    models = [
        load_and_cache_model(gcn_path, backend, use_jax=JAX_INSTALLED)
        for backend in backends
    ]
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
        "rbc_2_block_partial_ss.gcn",
        "full_nk_partial_ss.gcn",
    ],
    ids=["two_block", "full_nk"],
)
@pytest.mark.parametrize(
    "func", ["f_ss_error_grad", "f_ss_error_hess"], ids=["grad", "hess"]
)
def test_scipy_wrapped_functions_agree(gcn_path, func):
    backend_names = ["numpy", "numba", "pytensor"]
    models = [
        load_and_cache_model(gcn_path, backend, use_jax=JAX_INSTALLED)
        for backend in backend_names
    ]

    ss_variables = [x.to_ss() for x in models[0].variables]
    known_variables = list(models[0].f_ss(**models[0].parameters()).to_sympy().keys())

    vars_to_solve = [var for var in ss_variables if var not in known_variables]
    unknown_var_idx = np.array([x in vars_to_solve for x in ss_variables], dtype="bool")

    params = models[0].parameters().to_string()
    x0 = np.full(len(vars_to_solve), 0.8)

    vals = [
        scipy_wrapper(
            getattr(model, func),
            vars_to_solve,
            unknown_var_idx,
            unknown_var_idx,
            model.f_ss,
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


def test_linear_model():
    mod = load_and_cache_model("rbc_linearized.gcn", "numpy", use_jax=JAX_INSTALLED)
    params = mod.parameters()
    ss = mod.steady_state()

    assert all(x == 0 for x in ss.values())
    assert_allclose(mod.f_ss_error(**params, **ss), 0.0)

    assert not all(x == 0 for x in mod.f_ss(**mod.parameters()))


@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
@pytest.mark.parametrize(
    ("gcn_file", "expected_result"),
    [
        (
            "one_block_1_ss.gcn",
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
            "open_rbc.gcn",
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
            "full_nk.gcn",
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
def test_steady_state(backend: BACKENDS, gcn_file: str, expected_result: np.ndarray):
    n = len(expected_result)

    model = load_and_cache_model(gcn_file, backend, use_jax=JAX_INSTALLED)

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
    ["one_block_1_ss.gcn", "open_rbc.gcn", "full_nk.gcn"],
)
def test_model_gradient(backend, gcn_file):
    model = load_and_cache_model(gcn_file, backend, use_jax=JAX_INSTALLED)

    ss_result = model.steady_state()

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
    ["one_block_1_ss.gcn", "open_rbc.gcn", "full_nk.gcn", "rbc_with_excluded.gcn"],
)
@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
def test_numerical_steady_state(how: str, gcn_file: str, backend: BACKENDS):
    # TODO: I was hitting errors when the models were reused, something about the fixed values was breaking stuff.
    #  Need to track this bug down.
    model = load_and_cache_model(gcn_file, backend, use_jax=JAX_INSTALLED)
    analytic_res = model.steady_state(verbose=False, progressbar=False)
    analytic_values = np.array([analytic_res[x.to_ss().name] for x in model.variables])

    # Overwrite the f_ss function with None to trigger numerical optimization
    # Save it so we can put it back later, or else the cached model won't have a steady state function anymore
    f_ss = model.f_ss
    model.f_ss = None

    if gcn_file == "full_nk.gcn":
        fixed_values = {
            "shock_technology_ss": 1.0,
            "shock_preference_ss": 1.0,
            "pi_ss": 1.0,
            "pi_star_ss": 1.0,
            "pi_obj_ss": 1.0,
        }
    else:
        fixed_values = None

    numeric_res = model.steady_state(
        how=how,
        verbose=False,
        use_hess=True,
        use_hessp=False,
        optimizer_kwargs={
            "maxiter": 50_000,
            "method": "hybr" if how == "root" else "Newton-CG",
        },
        fixed_values=fixed_values,
        progressbar=False,
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
    file_path = "one_block_2_no_extra.gcn"
    model = load_and_cache_model(file_path, "numpy", use_jax=JAX_INSTALLED)

    res = model.steady_state(
        how="minimize",
        verbose=False,
        optimizer_kwargs={"method": "trust-constr", "options": {"maxiter": 100_000}},
        bounds={"alpha": (0.05, 0.7)},
        progressbar=False,
    )
    res = res.to_string()
    assert_allclose(res["L_ss"] / res["K_ss"], 0.36)


@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
def test_steady_state_with_parameter_updates(backend):
    file_path = "rbc_2_block_ss.gcn"
    model = load_and_cache_model(file_path, "numpy", use_jax=JAX_INSTALLED)

    rng = np.random.default_rng()
    delta = rng.beta(1, 1)
    beta = rng.beta(1, 1)
    ss_dict = model.steady_state(delta=delta, beta=beta)

    assert_allclose(ss_dict["r_ss"], (1 / beta - (1 - delta)))


@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
@pytest.mark.parametrize(
    "partial_file, analytic_file",
    [
        (
            "rbc_2_block_partial_ss.gcn",
            "rbc_2_block_ss.gcn",
        ),
        ("full_nk_partial_ss.gcn", "full_nk.gcn"),
    ],
)
def test_partially_analytical_steady_state(
    backend: BACKENDS, partial_file, analytic_file
):
    analytic_model = load_and_cache_model(analytic_file, backend, use_jax=JAX_INSTALLED)
    analytic_res = analytic_model.steady_state()
    analytic_values = np.array(list(analytic_res.values()))

    partial_model = load_and_cache_model(partial_file, backend, use_jax=JAX_INSTALLED)
    numeric_res = partial_model.steady_state(
        how="minimize",
        verbose=False,
        optimizer_kwargs={"method": "trust-ncg", "options": {"gtol": 1e-24}},
        progressbar=False,
        use_hessp=True,
        use_jac=True,
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
        ("one_block_1_ss.gcn", "one_block_ss"),
        ("rbc_2_block_ss.gcn", "two_block_ss"),
        ("full_nk.gcn", "full_nk"),
    ],
    ids=["one_block_ss", "two_block_ss", "full_nk"],
)
@pytest.mark.parametrize("backend", ["numba"], ids=["numba"])
def test_linearize(gcn_file, name, backend: BACKENDS):
    model = load_and_cache_model(gcn_file, backend, use_jax=JAX_INSTALLED)
    steady_state_dict = model.steady_state()
    outputs = model.linearize_model(
        loglin_negative_ss=True, steady_state=steady_state_dict
    )

    for mat_name, out in zip(["A", "B", "C", "D"], outputs):
        expected_out = expected_linearization_result[gcn_file][mat_name]
        assert_allclose(out, expected_out, atol=1e-8, err_msg=f"{mat_name} failed")


@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
def test_linearize_with_custom_params(backend):
    model = load_and_cache_model("one_block_1_ss.gcn", backend, use_jax=JAX_INSTALLED)
    params = model.parameters(rho=0.5)
    assert params["rho"] == 0.5

    # Use rho because d_shock_transiton/d_A = rho
    rho = np.random.beta(1, 1)
    A_idx = [x.base_name for x in model.variables].index("A")
    technology_eq_idx = next(
        i for i, eq in enumerate(model.equations) if model.shocks[0] in eq.atoms()
    )

    A, *_ = model.linearize_model(
        rho=rho,
        verbose=False,
        steady_state_kwargs={"verbose": False, "progressbar": False},
    )
    assert A[technology_eq_idx, A_idx] == rho


def test_invalid_solver_raises():
    file_path = "tests/_resources/test_gcns/one_block_1_ss.gcn"
    model = model_from_gcn(file_path, verbose=False)
    model.steady_state(verbose=False, progressbar=False)

    with pytest.raises(NotImplementedError):
        model.solve_model(
            solver="invalid_solver",
            steady_state_kwargs={"verbose": False, "progressbar": False},
            verbose=False,
        )


def test_bad_failure_argument_raises():
    file_path = "tests/_resources/test_gcns/pert_fails.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")

    with pytest.raises(ValueError):
        model.solve_model(
            solver="gensys",
            on_failure="raise",
            model_is_linear=True,  # TODO: This argument doesn't do anything yet
            steady_state_kwargs={"verbose": False, "progressbar": False},
            verbose=False,
        )


def test_gensys_fails_to_solve():
    file_path = "tests/_resources/test_gcns/pert_fails.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")

    with pytest.raises(GensysFailedException):
        model.solve_model(
            solver="gensys",
            on_failure="error",
            verbose=False,
            steady_state_kwargs={"verbose": False, "progressbar": False},
        )


def test_outputs_after_gensys_failure(caplog):
    file_path = "tests/_resources/test_gcns/pert_fails.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")
    T, R = model.solve_model(
        solver="gensys",
        on_failure="ignore",
        verbose=True,
        steady_state_kwargs={"verbose": False, "progressbar": False},
    )

    captured_message = caplog.messages[-1]
    assert captured_message == (
        "Gensys return codes: 1 0 2, with the following meaning:\n"
        "Solution exists, but is not unique."
    )
    assert T is None
    assert R is None


@pytest.mark.parametrize(
    "backend", ["numpy", "numba", "pytensor"], ids=["numpy", "numba", "pytensor"]
)
@pytest.mark.parametrize(
    "model_name, log_linearize",
    [
        ("one_block_1_ss", False),
        ("rbc_2_block_ss", False),
        ("full_nk", False),
        ("basic_rbc", False),
        ("basic_rbc", True),
    ],
    ids=lambda x: str(x),
)
def test_solve_matches_dynare(backend, model_name, log_linearize):
    gcn_file = model_name + ".gcn"
    model = load_and_cache_model(gcn_file, backend, use_jax=JAX_INSTALLED)
    T, R = model.solve_model(
        solver="gensys",
        verbose=False,
        log_linearize=log_linearize,
        steady_state_kwargs={"verbose": False, "progressbar": False},
    )

    if log_linearize:
        model_name = model_name + "_loglinear"

    dynare_T, dynare_R = load_dynare_outputs(model_name).values()

    T = matrix_to_dataframe(T, model).reindex_like(dynare_T)
    R = matrix_to_dataframe(R, model).reindex_like(dynare_R)

    assert_allclose(T[dynare_T.columns], dynare_T, atol=1e-5, rtol=1e-5)
    assert_allclose(R[dynare_R.columns], dynare_R, atol=1e-5, rtol=1e-5)


def test_outputs_after_pert_success(caplog):
    file_path = "tests/_resources/test_gcns/rbc_linearized.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")
    model.solve_model(
        solver="gensys",
        verbose=True,
        steady_state_kwargs={"verbose": False, "progressbar": False},
    )

    result_messages = caplog.messages[-2:]
    expected_messages = [
        "Norm of deterministic part: 0.000000000",
        "Norm of stochastic part:    0.000000000",
    ]

    for message, expected_message in zip(result_messages, expected_messages):
        assert message == expected_message


def test_bad_argument_to_bk_condition_raises():
    file_path = "tests/_resources/test_gcns/rbc_linearized.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")

    A, B, C, D = model.linearize_model()
    with pytest.raises(ValueError, match='Unknown return type "invalid_argument"'):
        check_bk_condition(A, B, C, D, return_value="invalid_argument", verbose=False)


def test_check_bk_condition():
    file_path = "tests/_resources/test_gcns/rbc_linearized.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")
    A, B, C, D = model.linearize_model()

    bk_df = check_bk_condition(A, B, C, D, return_value="dataframe", verbose=False)
    assert isinstance(bk_df, pd.DataFrame)

    assert_allclose(
        bk_df["Modulus"].values,
        np.abs(bk_df["Real"].values + bk_df["Imaginary"].values * 1j),
    )

    bk_res = check_bk_condition(A, B, C, D, return_value="bool", verbose=False)
    assert bk_res


def test_summarize_perturbation_solution():
    file_path = "tests/_resources/test_gcns/rbc_linearized.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")
    linear_system = [A, B, C, D] = model.linearize_model()
    policy_function = [T, R] = model.solve_model(solver="gensys", verbose=False)

    res = summarize_perturbation_solution(linear_system, policy_function, model)
    matrix_names = ["A", "B", "C", "D", "T", "R"]
    assert isinstance(res, xr.Dataset)
    assert all(name in res.data_vars for name in matrix_names)
    for matrix, name in zip([*linear_system, *policy_function], matrix_names):
        assert_allclose(res[name].to_numpy(), matrix)


def test_validate_shock_options():
    file_path = "tests/_resources/test_gcns/full_nk.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")
    T, R = model.solve_model(solver="gensys", verbose=False)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Exactly one of shock_std_dict, shock_cov_matrix, or shock_std should be provided. "
            "You passed 0."
        ),
    ):
        stationary_covariance_matrix(model, T, R)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Exactly one of shock_std_dict, shock_cov_matrix, or shock_std should be provided. "
            "You passed 2."
        ),
    ):
        stationary_covariance_matrix(
            model, T, R, shock_cov_matrix=np.eye(1), shock_std=0.1
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "If shock_std_dict is specified, it must give values for all shocks. "
            "The following shocks were not found among the provided keys: lol :)"
        ),
    ):
        stationary_covariance_matrix(model, T, R, shock_std_dict={"lol :)": 0.1})

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Incorrect covariance matrix shape. Expected (4, 4), found (2, 2)"
        ),
    ):
        stationary_covariance_matrix(model, T, R, shock_cov_matrix=np.eye(2))


def test_build_Q_matrix():
    file_path = "tests/_resources/test_gcns/full_nk.gcn"
    model = model_from_gcn(file_path, verbose=False, on_unused_parameters="ignore")
    shocks = model.shocks

    # From std
    Q = build_Q_matrix(
        model_shocks=shocks,
        shock_std=10,
    )

    assert_allclose(Q, np.eye(4) * 100)

    # From dictionary
    Q = build_Q_matrix(
        model_shocks=shocks,
        shock_std_dict={
            "epsilon_R": 0.1,
            "epsilon_pi": 0.2,
            "epsilon_Y": 0.3,
            "epsilon_preference": 0.4,
        },
    )
    # shocks get stored alphabetically (capitals first)
    expected_Q = np.diag(np.array([0.1, 0.3, 0.2, 0.4]) ** 2)
    assert_allclose(Q, expected_Q)

    # From cov
    L = np.random.normal(size=(4, 4))
    cov = L @ L.T

    Q = build_Q_matrix(
        model_shocks=shocks,
        shock_cov_matrix=cov,
    )

    assert_allclose(Q, cov)


def test_build_Q_matrix_from_dict():
    file_path = "full_nk.gcn"
    model = load_and_cache_model(file_path, "numpy", use_jax=JAX_INSTALLED)
    shocks = model.shocks

    L = np.random.normal(size=(4, 4))
    cov = L @ L.T

    Q = build_Q_matrix(
        model_shocks=shocks,
        shock_cov_matrix=cov,
    )

    assert_allclose(Q, cov)


def test_compute_stationary_covariance_warns_on_partial_specification(caplog):
    model = load_and_cache_model("rbc_linearized.gcn", "numpy", use_jax=JAX_INSTALLED)
    T, R = model.solve_model(solver="gensys", verbose=False)

    stationary_covariance_matrix(model, T, shock_std=0.1, verbose=False)
    messages = caplog.messages
    assert messages[-1].startswith("Passing only one of T or R will still trigger")


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "open_rbc.gcn",
        "full_nk.gcn",
        "rbc_linearized.gcn",
    ],
)
def test_compute_stationary_covariance(caplog, gcn_file):
    model = load_and_cache_model(gcn_file, backend="numpy", use_jax=JAX_INSTALLED)
    T, R = model.solve_model(solver="gensys", verbose=False)
    n_variables, n_shocks = R.shape

    Sigma = stationary_covariance_matrix(model, T, R, shock_std=0.1, return_df=False)
    assert len(caplog.messages) == 0
    assert Sigma.shape == (n_variables, n_variables)

    assert_allclose(Sigma, Sigma.T, atol=1e-8)
    assert all(x > 0 for x in np.diagonal(Sigma))

    # Check for PSD by getting the closest PSD matrix (setting negative eigenvalues to zero) then
    # checking if the result is close to the original.
    eigvals, eigvecs = np.linalg.eig(Sigma)
    eigvals = np.where(eigvals < 0, 0, eigvals)
    Sigma_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    assert_allclose(Sigma, Sigma_psd, atol=1e-8)


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "open_rbc.gcn",
        "full_nk.gcn",
        "rbc_linearized.gcn",
    ],
)
def test_autocovariance_matrix(caplog, gcn_file):
    model = load_and_cache_model(gcn_file, backend="numpy", use_jax=JAX_INSTALLED)

    shocks = model.shocks
    shock_eqs = [eq for eq in model.equations if any(s in eq.atoms() for s in shocks)]

    for eq in shock_eqs:
        atoms = eq.atoms()
        shock = next(x for x in atoms if x in shocks)
        if shock.base_name in ["epsilon_R", "epsilon_pi"]:
            # These aren't a normal AR(1) shocks, so we skip them
            continue

        state = next(x for x in atoms if x in model.variables)
        state_idx = model.variables.index(state)

        rho = next(x for x in atoms if x in model.params)
        rho_value = np.random.beta(10, 1)

        # The autocorrelation of the AR(1) states decay at rate rho ** t
        # Other autocovarainces are more complex, but this one is easy to check
        autocorr = autocorrelation_matrix(
            model,
            shock_std=0.1,
            solver="gensys",
            verbose=False,
            return_xr=False,
            **{rho.name: rho_value},
        )

        assert_allclose(
            autocorr[:, state_idx, state_idx],
            rho_value ** np.arange(10),
            atol=1e-8,
            rtol=1e-8,
            err_msg=f"Error computing {state} autocovariance in {gcn_file}",
        )


def setup_cov_arguments(argument, n_shocks, model):
    shock_std = None
    shock_dict = None
    shock_cov_matrix = None
    if argument == "shock_std":
        shock_std = 0.1
    elif argument == "shock_std_dict":
        shock_dict = {shock.base_name: 0.1 for shock in model.shocks}
    elif argument == "shock_cov_matrix":
        shock_cov_matrix = np.eye(n_shocks) * 0.1**2

    return shock_std, shock_dict, shock_cov_matrix


@pytest.mark.parametrize(
    "shock_size",
    [
        0.1,
        np.array([0.1, 0.1]),
        {"epsilon_A": 0.1, "epsilon_B": 0.1},
        {"epsilon_B": 0.1},
    ],
    ids=["single_float", "array", "dict", "partial_dict"],
)
@pytest.mark.parametrize(
    "return_individual_shocks", [True, False], ids=["individual_shocks", "joint_shocks"]
)
def test_irf_from_shock_size(shock_size, return_individual_shocks):
    file_path = "one_block_1_ss_2shock.gcn"
    model = load_and_cache_model(file_path, backend="numpy", use_jax=JAX_INSTALLED)
    T, R = model.solve_model(solver="gensys", verbose=False)
    n_variables, n_shocks = R.shape

    irf = impulse_response_function(
        model,
        T,
        R,
        simulation_length=1000,
        shock_size=shock_size,
        return_individual_shocks=return_individual_shocks,
    )

    assert "time" in irf.coords
    assert "variable" in irf.coords

    if return_individual_shocks:
        assert "shock" in irf.coords
        if isinstance(shock_size, dict):
            assert set(irf.coords["shock"].values) == set(shock_size.keys())
    else:
        assert "shock" not in irf.coords

    assert len(irf.coords["time"]) == 1000
    assert len(irf.coords["variable"]) == n_variables

    # After 1000 steps the shocks should have mostly died out
    assert np.all(np.abs(irf.isel(time=-1).values) < 1e-3)

    n_test_shocks = 1 if isinstance(shock_size, float | int) else len(shock_size)
    if (n_shocks > 1) and (n_test_shocks > 1) and return_individual_shocks:
        assert not np.allclose(
            irf.sel(shock="epsilon_A").values, irf.sel(shock="epsilon_B").values
        )


@pytest.mark.parametrize(
    "return_individual_shocks", [True, False], ids=["individual_shocks", "joint_shocks"]
)
@pytest.mark.parametrize("n_shocks", [1, 2], ids=["single_shock", "two_shocks"])
def test_irf_from_trajectory(return_individual_shocks, n_shocks):
    file_path = "one_block_1_ss_2shock.gcn"
    model = load_and_cache_model(file_path, backend="numpy", use_jax=JAX_INSTALLED)
    T, R = model.solve_model(solver="gensys", verbose=False)
    n_variables, n_shocks = R.shape

    shock_trajectory = np.zeros((1000, n_shocks))
    for i in range(n_shocks):
        shock_trajectory[0, i] = 0.1

    irf = impulse_response_function(
        model,
        T,
        R,
        simulation_length=1000,
        shock_trajectory=shock_trajectory,
        return_individual_shocks=return_individual_shocks,
    )

    assert "time" in irf.coords
    assert "variable" in irf.coords

    if return_individual_shocks:
        assert "shock" in irf.coords
    else:
        assert "shock" not in irf.coords

    assert len(irf.coords["time"]) == 1000
    assert len(irf.coords["variable"]) == n_variables
    assert np.all(np.abs(irf.isel(time=-1).values) < 1e-3)

    if (n_shocks == 2) and return_individual_shocks:
        assert not np.allclose(
            irf.sel(shock="epsilon_A").values, irf.sel(shock="epsilon_B").values
        )


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "open_rbc.gcn",
        "full_nk.gcn",
    ],
)
@pytest.mark.parametrize(
    "argument", ["shock_std", "shock_std_dict", "shock_cov_matrix"]
)
def test_simulate(gcn_file, argument):
    model = load_and_cache_model(gcn_file, backend="numpy", use_jax=JAX_INSTALLED)
    T, R = model.solve_model(solver="gensys", verbose=False)
    n_variables, n_shocks = R.shape

    n_simulations = 3000
    simulation_length = 2000

    shock_std, shock_std_dict, shock_cov_matrix = setup_cov_arguments(
        argument, n_shocks, model
    )

    data = simulate(
        model,
        T,
        R,
        simulation_length=simulation_length,
        n_simulations=n_simulations,
        shock_std=shock_std,
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
    )

    assert data.shape == (n_simulations, simulation_length, n_variables)

    # Check that the simulated covariance matrix is at least strong correlated with the stationary covariance matrix
    # across many trajectories
    Sigma = stationary_covariance_matrix(model, T, R, shock_std=0.1, return_df=False)
    sigma = np.cov(data.isel(time=-1).values.T)

    corr = np.corrcoef(np.r_[Sigma.ravel(), sigma.ravel()])
    assert abs(corr) > 0.99

    assert_allclose(np.diag(Sigma), np.diag(sigma), rtol=0.1)


def test_objective_with_complex_discount_factor():
    gcn_file = "rbc_firm_capital.gcn"
    model = load_and_cache_model(gcn_file, backend="numpy", use_jax=JAX_INSTALLED)

    ss_res = model.steady_state(
        verbose=False, how="minimize", optimizer_kwargs={"method": "Newton-CG"}
    )
    assert ss_res.success

    bk_success = check_bk_condition(
        *model.linearize_model(steady_state=ss_res),
        return_value="bool",
        verbose=False,
    )
    assert bk_success

    gcn_file = "rbc_firm_capital_comparison.gcn"
    model_2 = load_and_cache_model(gcn_file, backend="numpy", use_jax=JAX_INSTALLED)

    ss_res_2 = model_2.steady_state(verbose=False)
    assert ss_res_2.success

    assert_allclose(ss_res["Y_ss"], ss_res_2["Y_ss"], rtol=1e-8, atol=1e-8)
    assert_allclose(ss_res["K_ss"], ss_res_2["K_ss"], rtol=1e-8, atol=1e-8)
    assert_allclose(ss_res["L_ss"], ss_res_2["L_ss"], rtol=1e-8, atol=1e-8)
    assert_allclose(ss_res["I_ss"], ss_res_2["I_ss"], rtol=1e-8, atol=1e-8)
