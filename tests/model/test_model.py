import itertools
import re

import numdifftools as nd
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from numpy.testing import assert_allclose
from pytensor.graph.traversal import explicit_graph_inputs

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import GensysFailedException, ModelUnknownParameterError
from gEconpy.model.build import model_from_gcn
from gEconpy.model.compile import compile_for_scipy, make_cache_key
from gEconpy.model.model import DROrder
from gEconpy.model.perturbation import (
    check_bk_condition,
    compute_bk_eigenvalues,
    compute_bk_eigenvalues_pt,
)
from gEconpy.model.simulate import impulse_response_function, simulate
from gEconpy.model.statistics import (
    autocorrelation_matrix,
    autocovariance_matrix,
    build_Q_matrix,
    eigenvalue_sensitivity,
    matrix_to_dataframe,
    stationary_covariance_matrix,
    summarize_perturbation_solution,
)
from gEconpy.model.statistics.validation import _maybe_linearize_model
from gEconpy.model.steady_state import _ss_residual_to_pytensor, build_minimize_graphs, build_root_graphs
from gEconpy.utilities import safe_to_ss
from tests._resources.cache_compiled_models import load_and_cache_model
from tests._resources.expected_matrices import expected_linearization_result
from tests._resources.load_dynare import load_dynare_outputs
from tests.conftest import TEST_GCNS


@pytest.fixture
def rng():
    return np.random.default_rng(seed=1234)


def _model_without_analytic_steady_state(gcn_file):
    """Build a fresh model and strip its analytic steady state so every variable must be solved numerically."""
    model = model_from_gcn(TEST_GCNS / gcn_file, verbose=False, mode="FAST_RUN")
    model._ss_solution_dict = SymbolDictionary()
    model._f_ss = None
    model._equation_tensors = None
    return model


@pytest.mark.parametrize(
    "gcn_path",
    [
        "one_block_1_dist.gcn",
        "one_block_1_ss.gcn",
        pytest.param("full_nk.gcn", marks=pytest.mark.include_nk),
    ],
    ids=["one_block_prior", "one_block_ss", "full_nk"],
)
def test_model_parameters(gcn_path: str):
    model = load_and_cache_model(gcn_path)

    params = model.parameters()
    assert all(params[k] == model._default_params[k] for k in model._default_params)
    assert all(isinstance(v, float) for v in params.values())

    old_params = model._default_params.copy()
    params = model.parameters(beta=0.5)
    assert params["beta"] == 0.5
    assert model._default_params["beta"] == old_params["beta"]


def test_deterministic_model_parameters():
    model = load_and_cache_model("one_block_2.gcn")

    params = model.parameters()
    assert_allclose(params["beta"], 1 / 1.01)

    params = model.parameters(theta=0.9)
    assert params["theta"] == 0.9
    assert_allclose(params["zeta"], -np.log(0.9))

    # Deterministic names are ignored, so the output of one call can be passed back into the next.
    params = model.parameters(theta=0.9, zeta=100.0)
    assert_allclose(params["zeta"], -np.log(0.9))
    assert model.parameters(**params) == params


def test_unknown_parameter_update_raises():
    model = load_and_cache_model("one_block_1_ss.gcn")

    with pytest.raises(ModelUnknownParameterError, match="do not exist in the model: not_a_param"):
        model.parameters(not_a_param=1.0)


def test_get_returns_symbols_and_names_a_close_match():
    model = load_and_cache_model("one_block_1_ss.gcn")

    assert model.get("K") == model.variables[[v.base_name for v in model.variables].index("K")]
    assert model.get("K_ss") == model.get("K").to_ss()
    assert model.get("alpha") in model.params

    with pytest.raises(IndexError, match=re.escape("Did not find Kk among model objects. Did you mean K")):
        model.get("Kk")


def test_linear_model():
    mod = load_and_cache_model("rbc_linearized.gcn")
    ss = mod.steady_state()

    assert all(x == 0 for x in ss.values())
    assert ss.success

    # f_ss holds the level values of the underlying nonlinear model, which are not zero.
    assert not all(x == 0 for x in mod.f_ss(**mod.parameters()))


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
        pytest.param(
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
            marks=pytest.mark.include_nk,
        ),
    ],
    ids=["one_block", "open_rbc", "nk"],
)
def test_steady_state(gcn_file: str, expected_result: dict[str, float]):
    model = load_and_cache_model(gcn_file)

    params = model.parameters()
    ss_dict = model.f_ss(**params)
    ss = np.array(list(ss_dict.values()))
    expected_ss = np.array([expected_result[var] for var in ss_dict.to_string()])

    assert_allclose(ss, expected_ss)
    assert_allclose(model._evaluate_steady_state(), np.zeros(len(expected_result)), atol=1e-8)

    ss_result = model.steady_state(verbose=False, progressbar=False)
    assert ss_result.success


def _compile_ss_derivative_funcs(gcn_file):
    # These graphs are evaluated only a handful of times, so FAST_COMPILE avoids paying for C compilation.
    model = load_and_cache_model(gcn_file)
    ss_result = model.steady_state()

    equations, cache = _ss_residual_to_pytensor(
        model._steady_state_equations,
        SymbolDictionary(),
        model._variables,
        model._param_dict,
        model._deterministic_dict,
        model._calib_dict,
    )

    resid = pt.stack(equations)
    resid_inputs = set(explicit_graph_inputs(resid))
    ss_symbols = [safe_to_ss(v) for v in list(model._variables) + list(model._calib_dict.to_sympy().keys())]
    ss_nodes = []
    for symbol in ss_symbols:
        node = cache.get(make_cache_key(symbol.name, type(symbol)))
        if node is not None and node in resid_inputs:
            ss_nodes.append(node)

    error_graph, grad_graph, hess_graph, hessp_graph, hessp_p = build_minimize_graphs(
        equations,
        ss_nodes,
        error_func=model._error_func,
        use_jac=True,
        use_hess=True,
        use_hessp=True,
    )
    _, jac_graph = build_root_graphs(equations, ss_nodes, use_jac=True)

    compiled = {
        name: compile_for_scipy(graph, mode="FAST_COMPILE")
        for name, graph in [
            ("resid", resid),
            ("jac", jac_graph),
            ("error", error_graph),
            ("grad", grad_graph),
            ("hess", hess_graph),
            ("hessp", hessp_graph),
        ]
    }
    compiled["hessp_direction_name"] = hessp_p.name
    return model, ss_result, compiled


def test_ss_derivatives_match_numeric():
    # Derivative-graph construction does not depend on the model, so one small model is enough.
    model, ss_result, f = _compile_ss_derivative_funcs("one_block_1_ss.gcn")
    params = model.parameters()

    assert_allclose(f["grad"](**ss_result, **params), 0.0, rtol=1e-12, atol=1e-12)

    perturbed_point = {k: np.float64(0.8) for k in ss_result}
    test_point = np.array(list(perturbed_point.values()))

    def at(x):
        return dict(zip(perturbed_point, x, strict=True))

    grad = np.asarray(f["grad"](**perturbed_point, **params))
    numeric_grad = nd.Gradient(lambda x: float(np.asarray(f["error"](**at(x), **params))))(test_point)
    assert_allclose(grad, numeric_grad, rtol=1e-8, atol=1e-8)

    hess = np.asarray(f["hess"](**perturbed_point, **params))
    numeric_hess = nd.Hessian(lambda x: float(np.asarray(f["error"](**at(x), **params))))(test_point)
    assert_allclose(hess, numeric_hess, rtol=1e-8, atol=1e-8)

    direction = np.arange(1.0, len(test_point) + 1)
    hessp = np.asarray(f["hessp"](**perturbed_point, **params, **{f["hessp_direction_name"]: direction}))
    assert_allclose(hessp, numeric_hess @ direction, rtol=1e-8, atol=1e-8)

    jac = np.asarray(f["jac"](**perturbed_point, **params))
    numeric_jac = nd.Jacobian(lambda x: np.asarray(f["resid"](**at(x), **params)).ravel())(test_point)
    assert_allclose(jac, numeric_jac, rtol=1e-8, atol=1e-8)


@pytest.mark.include_nk
def test_ss_derivative_graphs_compile():
    # Numeric correctness is covered by test_ss_derivatives_match_numeric. This runs the same pipeline on the largest
    # model to catch build and compile regressions on a hard case.
    model, ss_result, f = _compile_ss_derivative_funcs("full_nk.gcn")
    params = model.parameters()

    assert_allclose(f["grad"](**ss_result, **params), 0.0, rtol=1e-12, atol=1e-12)

    for name in ("resid", "jac", "hess"):
        assert np.all(np.isfinite(np.asarray(f[name](**ss_result, **params))))


@pytest.mark.parametrize(
    ("how", "optimizer_kwargs"),
    [("root", {"maxiter": 50_000, "method": "hybr", "options": {"xtol": 1e-12}}), ("minimize", {})],
    ids=["root", "minimize"],
)
@pytest.mark.parametrize(
    ("gcn_file", "fixed_values"),
    [
        ("one_block_1_ss.gcn", None),
        ("open_rbc.gcn", None),
        ("rbc_with_excluded.gcn", None),
        pytest.param(
            "full_nk.gcn",
            {
                "shock_technology_ss": 1.0,
                "shock_preference_ss": 1.0,
                "pi_ss": 1.0,
                "pi_star_ss": 1.0,
                "pi_obj_ss": 1.0,
            },
            marks=pytest.mark.include_nk,
        ),
    ],
    ids=["one_block_ss", "open_rbc", "rbc_with_excluded", "full_nk"],
)
def test_numerical_steady_state(how, optimizer_kwargs, gcn_file, fixed_values):
    analytic_res = load_and_cache_model(gcn_file).steady_state(verbose=False, progressbar=False)

    model = _model_without_analytic_steady_state(gcn_file)
    numeric_res = model.steady_state(
        how=how,
        verbose=False,
        optimizer_kwargs=optimizer_kwargs,
        fixed_values=fixed_values,
        progressbar=False,
    )

    analytic_values = np.array([analytic_res[x.to_ss().name] for x in model.variables])
    numeric_values = np.array([numeric_res[x.to_ss().name] for x in model.variables])
    residuals = model.evaluate_residual(numeric_res, model.parameters())

    assert numeric_res.success
    assert_allclose(numeric_values, analytic_values, atol=1e-4)
    assert_allclose(residuals, 0.0, atol=1e-6)


def test_numerical_steady_state_with_calibrated_params():
    model = load_and_cache_model("one_block_2_no_extra.gcn")

    res = model.steady_state(
        how="minimize",
        verbose=False,
        use_hess=False,
        use_hessp=False,
        optimizer_kwargs={"method": "L-BFGS-B", "options": {"maxiter": 100_000}},
        bounds={"alpha": (0.05, 0.7)},
        progressbar=False,
    )
    res = res.to_string()
    assert_allclose(res["L_ss"] / res["K_ss"], 0.36)


def test_steady_state_with_parameter_updates(rng):
    model = load_and_cache_model("rbc_2_block_ss.gcn")

    delta = rng.beta(1, 1)
    beta = rng.beta(1, 1)
    ss_dict = model.steady_state(delta=delta, beta=beta)

    assert_allclose(ss_dict["r_ss"], (1 / beta - (1 - delta)))


@pytest.mark.parametrize(
    "partial_file",
    [
        "rbc_2_block_partial_ss.gcn",
        pytest.param("full_nk_partial_ss.gcn", marks=pytest.mark.include_nk),
    ],
)
def test_partially_analytical_steady_state(partial_file):
    partial_model = load_and_cache_model(partial_file)
    numeric_res = partial_model.steady_state(
        how="minimize",
        verbose=False,
        optimizer_kwargs={"method": "L-BFGS-B"},
        progressbar=False,
        use_hess=False,
        use_hessp=False,
        use_jac=True,
    )

    resid = partial_model.evaluate_residual(numeric_res.to_string(), partial_model.parameters())
    assert_allclose(resid, 0, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "rbc_2_block_ss.gcn",
        pytest.param("full_nk.gcn", marks=pytest.mark.include_nk),
    ],
    ids=["one_block_ss", "two_block_ss", "full_nk"],
)
def test_linearize(gcn_file):
    model = load_and_cache_model(gcn_file)
    steady_state_dict = model.steady_state()
    outputs = model.linearize_model(steady_state=steady_state_dict)

    for mat_name, out in zip(["A", "B", "C", "D"], outputs, strict=True):
        expected_out = expected_linearization_result[gcn_file][mat_name]
        assert_allclose(out, expected_out, atol=1e-8, err_msg=f"{mat_name} failed")


def test_linearize_with_custom_params(rng):
    model = load_and_cache_model("one_block_1_ss.gcn")

    # The technology process is A[t] = rho * A[t-1] + eps, so d(equation)/d(A[t-1]) is rho.
    rho = rng.beta(1, 1)
    A_idx = [x.base_name for x in model.variables].index("A")
    technology_eq_idx = next(i for i, eq in enumerate(model.equations) if model.shocks[0] in eq.atoms())

    A, *_ = model.linearize_model(
        rho=rho,
        verbose=False,
        steady_state_kwargs={"verbose": False, "progressbar": False},
    )
    assert A[technology_eq_idx, A_idx] == rho


def test_linearize_steady_state_kwargs_override_verbose(caplog):
    model = _model_without_analytic_steady_state("one_block_1_ss.gcn")

    model.linearize_model(verbose=True, steady_state_kwargs={"verbose": False, "progressbar": False})

    assert not any(message.startswith("Steady state") for message in caplog.messages)


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "rbc_2_block_ss.gcn",
        pytest.param("full_nk.gcn", marks=pytest.mark.include_nk),
    ],
    ids=["one_block_ss", "two_block_ss", "full_nk"],
)
def test_symbolic_linearization_graphs_evaluate_to_permuted_linearization(gcn_file):
    model = load_and_cache_model(gcn_file)
    steady_state = model.steady_state(verbose=False, progressbar=False)
    params = model.parameters()

    jacobians, ss_nodes, param_nodes, eq_order, var_order = model.symbolic_linearization(verbose=False)
    f = pytensor.function([*ss_nodes, *param_nodes], jacobians, mode="FAST_COMPILE", on_unused_input="ignore")

    ss_inputs = [steady_state[f"{v.base_name}_ss"] for v in model.variables]
    A, B, C, D = f(*ss_inputs, *[params[node.name] for node in param_nodes])

    # The graphs carry rows in eq_order and columns in var_order. Undoing both recovers linearize_model's layout.
    inv_eq, inv_var = np.argsort(eq_order), np.argsort(var_order)
    unpermuted = [A[inv_eq][:, inv_var], B[inv_eq][:, inv_var], C[inv_eq][:, inv_var], D[inv_eq]]
    expected = model.linearize_model(steady_state=steady_state, verbose=False)

    for name, actual, reference in zip("ABCD", unpermuted, expected, strict=True):
        assert_allclose(actual, reference, atol=1e-8, err_msg=name)


def test_symbolic_linearization_caches():
    model = load_and_cache_model("one_block_1_ss.gcn")
    jac1, *_ = model.symbolic_linearization(verbose=False)
    jac2, *_ = model.symbolic_linearization(verbose=False)

    for a, b in zip(jac1, jac2, strict=True):
        assert a is b


@pytest.mark.parametrize("method_name", ["linearize_model", "symbolic_linearization"])
def test_second_order_linearization_raises(method_name):
    model = load_and_cache_model("one_block_1_ss.gcn")

    with pytest.raises(NotImplementedError, match="Only first order linearization is currently supported"):
        getattr(model, method_name)(order=2, verbose=False)


def test_dr_order_groups_variables_and_equations_by_time_shift():
    static, pred, mixed, fwd = (TimeAwareSymbol(name, 0) for name in ["static", "pred", "mixed", "fwd"])
    equations = [
        fwd.set_t(1) - mixed,
        static - pred,
        pred - 0.5 * pred.set_t(-1),
        mixed.set_t(1) - mixed.set_t(-1) + static,
    ]

    order = DROrder.from_model([fwd, mixed, pred, static], equations)

    assert order.var_order.tolist() == [3, 2, 1, 0]
    assert order.eq_order.tolist() == [1, 2, 0, 3]
    assert (order.n_static_var, order.n_pred_only_var, order.n_mixed_var, order.n_forward_only_var) == (1, 1, 1, 1)
    assert (order.n_static_eq, order.n_lag_only_eq, order.n_lead_only_eq, order.n_both_eq) == (1, 1, 1, 1)
    assert order.var_order[order.inv_var_order].tolist() == [0, 1, 2, 3]
    assert order.eq_order[order.inv_eq_order].tolist() == [0, 1, 2, 3]


def test_invalid_solver_raises():
    model = model_from_gcn(TEST_GCNS / "one_block_1_ss.gcn", verbose=False)
    model.steady_state(verbose=False, progressbar=False)

    with pytest.raises(NotImplementedError):
        model.solve_model(
            solver="invalid_solver",
            steady_state_kwargs={"verbose": False, "progressbar": False},
            verbose=False,
        )


def test_bad_failure_argument_raises():
    model = model_from_gcn(TEST_GCNS / "pert_fails.gcn", verbose=False, on_unused_parameters="ignore")

    with pytest.raises(ValueError, match='on_failure must be one of "error" or "ignore"'):
        model.solve_model(
            solver="gensys",
            on_failure="raise",
            steady_state_kwargs={"verbose": False, "progressbar": False},
            verbose=False,
        )


@pytest.mark.parametrize(
    ("solver", "match"),
    [("gensys", "Gensys return codes"), ("cycle_reduction", "^Iteration on all matrices failed to converge$")],
)
def test_unsolvable_model_raises_with_the_solver_message(solver, match):
    model = model_from_gcn(TEST_GCNS / "pert_fails.gcn", verbose=False, on_unused_parameters="ignore")

    with pytest.raises(GensysFailedException, match=match):
        model.solve_model(
            solver=solver,
            on_failure="error",
            verbose=False,
            steady_state_kwargs={"verbose": False, "progressbar": False},
        )


def test_outputs_after_gensys_failure(caplog):
    model = model_from_gcn(TEST_GCNS / "pert_fails.gcn", verbose=False, on_unused_parameters="ignore")
    T, R = model.solve_model(
        solver="gensys",
        on_failure="ignore",
        verbose=True,
        steady_state_kwargs={"verbose": False, "progressbar": False},
    )

    assert caplog.messages[-1] == (
        "Gensys return codes: 1 0 2, with the following meaning:\nSolution exists, but is not unique."
    )
    assert T is None
    assert R is None


@pytest.mark.parametrize(
    "model_name, log_linearize",
    [
        ("one_block_1_ss", False),
        ("rbc_2_block_ss", False),
        pytest.param("full_nk", False, marks=pytest.mark.include_nk),
        ("basic_rbc", False),
        ("basic_rbc", True),
    ],
    ids=str,
)
def test_solve_matches_dynare(model_name, log_linearize):
    model = load_and_cache_model(model_name + ".gcn")
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
    model = model_from_gcn(TEST_GCNS / "rbc_linearized.gcn", verbose=False, on_unused_parameters="ignore")
    model.solve_model(
        solver="gensys",
        verbose=True,
        steady_state_kwargs={"verbose": False, "progressbar": False},
    )

    assert caplog.messages[-2:] == [
        "Norm of deterministic part: 0.000000000",
        "Norm of stochastic part:    0.000000000",
    ]


def test_bad_argument_to_bk_condition_raises():
    model = load_and_cache_model("rbc_linearized.gcn")

    A, B, C, D = model.linearize_model()
    with pytest.raises(ValueError, match='"invalid_argument"'):
        check_bk_condition(A, B, C, D, return_value="invalid_argument", verbose=False)


def test_check_bk_condition():
    model = load_and_cache_model("rbc_linearized.gcn")
    A, B, C, D = model.linearize_model()

    bk_df = check_bk_condition(A, B, C, D, return_value="dataframe", verbose=False)
    assert isinstance(bk_df, pd.DataFrame)
    assert_allclose(
        bk_df["Modulus"].values,
        np.abs(bk_df["Real"].values + bk_df["Imaginary"].values * 1j),
    )

    assert check_bk_condition(A, B, C, D, return_value="bool", verbose=False)


def test_compute_bk_eigenvalues():
    model = load_and_cache_model("rbc_linearized.gcn")
    A, B, C, D = model.linearize_model()

    eigvals_real, eigvals_imag, n_forward = compute_bk_eigenvalues(A, B, C, D)

    modulus = np.sqrt(eigvals_real**2 + eigvals_imag**2)
    n_unstable = (modulus > 1).sum()

    assert n_forward > 0
    assert n_forward == n_unstable
    assert np.all(np.diff(modulus) >= -1e-12)


def test_compute_bk_eigenvalues_pt():
    model = load_and_cache_model("rbc_linearized.gcn")
    A_np, B_np, C_np, D_np = model.linearize_model()

    eigvals_real_np, eigvals_imag_np, _n_forward = compute_bk_eigenvalues(A_np, B_np, C_np, D_np)
    lead_var_idx = np.where(np.abs(C_np).sum(axis=0) > 1e-8)[0]

    A_pt, B_pt, C_pt, D_pt = (pt.as_tensor_variable(M) for M in (A_np, B_np, C_np, D_np))
    re_pt, im_pt = compute_bk_eigenvalues_pt(A_pt, B_pt, C_pt, D_pt, lead_var_idx)
    re_val, im_val = re_pt.eval(), im_pt.eval()

    modulus_np = np.sqrt(eigvals_real_np**2 + eigvals_imag_np**2)
    modulus_pt = np.sqrt(re_val**2 + im_val**2)

    # QZ and solve-plus-eig may return different infinite or spurious eigenvalues, but the count of unstable ones
    # must agree.
    assert (modulus_np > 1).sum() == (modulus_pt > 1).sum()


def test_eigenvalue_sensitivity():
    model = load_and_cache_model("basic_rbc.gcn")
    A, B, C, D = model.linearize_model(verbose=False)

    re_np, im_np, _ = compute_bk_eigenvalues(A, B, C, D)
    mod_np = np.sqrt(re_np**2 + im_np**2)

    ds = eigenvalue_sensitivity(model, verbose=False)
    mod_pt = ds.eigenvalues.sel(component="modulus").values

    finite_np = mod_np[(mod_np > 1e-6) & (mod_np < 1e6)]
    finite_pt = mod_pt[(mod_pt > 1e-6) & (mod_pt < 1e6)]

    # QZ (numpy reference) and solve-plus-eig (sensitivity path) agree only to about 5 significant figures by
    # construction, and the QZ ordering of the clustered eigenvalues wobbles run to run. rtol=1e-5 sits exactly on
    # that boundary and flakes, so do not tighten.
    assert_allclose(np.sort(finite_np), np.sort(finite_pt), rtol=1e-4)

    grad_mags = np.sqrt(ds.gradients.sel(part="real").values ** 2 + ds.gradients.sel(part="imaginary").values ** 2)
    assert grad_mags.max() > 1e-10


def test_summarize_perturbation_solution():
    model = load_and_cache_model("rbc_linearized.gcn")
    linear_system = model.linearize_model()
    policy_function = model.solve_model(solver="gensys", verbose=False)

    res = summarize_perturbation_solution(linear_system, policy_function, model)
    matrix_names = ["A", "B", "C", "D", "T", "R"]
    assert isinstance(res, xr.Dataset)
    assert all(name in res.data_vars for name in matrix_names)
    for matrix, name in zip([*linear_system, *policy_function], matrix_names, strict=True):
        assert_allclose(res[name].to_numpy(), matrix)


@pytest.mark.parametrize(
    "shock_kwargs, expected_msg",
    [
        ({}, "Exactly one of shock_std_dict, shock_cov_matrix, or shock_std should be provided. You passed 0."),
        (
            {"shock_cov_matrix": np.eye(1), "shock_std": 0.1},
            "Exactly one of shock_std_dict, shock_cov_matrix, or shock_std should be provided. You passed 2.",
        ),
        (
            {"shock_std_dict": {"lol :)": 0.1}},
            "Unexpected shocks in shock_std_dict. The following names were not found among the model shocks: lol :)",
        ),
        (
            {"shock_std_dict": {"epsilon_R": 0.1, "epsilon_pi": 0.1}},
            "If shock_std_dict is specified, it must give values for all shocks. The following shocks were not found "
            "among the provided keys: epsilon_Y, epsilon_preference",
        ),
        ({"shock_cov_matrix": np.eye(2)}, "Incorrect covariance matrix shape. Expected (4, 4), found (2, 2)"),
        ({"shock_std": [0.1, 0.2]}, "Length of shock_std (2) does not match the number of shocks (4)"),
        ({"shock_std": [0.1, 0.2, 0.0, 0.4]}, "Shock standard deviations must be positive"),
        ({"shock_std": -0.1}, "Shock standard deviation must be positive"),
    ],
    ids=[
        "none_given",
        "two_given",
        "unknown_shock_name",
        "missing_shock_name",
        "wrong_cov_shape",
        "wrong_std_length",
        "non_positive_std_list",
        "negative_std_scalar",
    ],
)
def test_validate_shock_options(shock_kwargs, expected_msg):
    model = load_and_cache_model("full_nk.gcn")
    T, R = model.solve_model(solver="gensys", verbose=False)

    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        stationary_covariance_matrix(model, T, R, **shock_kwargs)


def test_build_Q_matrix(rng):
    model = load_and_cache_model("full_nk.gcn")
    shocks = model.shocks

    Q = build_Q_matrix(model_shocks=shocks, shock_std=10)
    assert_allclose(Q, np.eye(4) * 100)

    Q = build_Q_matrix(
        model_shocks=shocks,
        shock_std_dict={
            "epsilon_R": 0.1,
            "epsilon_pi": 0.2,
            "epsilon_Y": 0.3,
            "epsilon_preference": 0.4,
        },
    )
    # Shocks are stored in sorted order, capitals first.
    expected_Q = np.diag(np.array([0.1, 0.3, 0.2, 0.4]) ** 2)
    assert_allclose(Q, expected_Q)

    L = rng.normal(size=(4, 4))
    cov = L @ L.T
    Q = build_Q_matrix(model_shocks=shocks, shock_cov_matrix=cov)
    assert_allclose(Q, cov)


def test_build_Q_matrix_accepts_list_of_stds():
    model = load_and_cache_model("full_nk.gcn")
    stds = [0.1, 0.2, 0.3, 0.4]

    Q = build_Q_matrix(model_shocks=model.shocks, shock_std=stds)
    assert_allclose(Q, np.diag(np.array(stds) ** 2))


@pytest.mark.parametrize("verbose", [True, False], ids=["verbose", "quiet"])
def test_maybe_linearize_model_relinearizes_partial_input(caplog, verbose):
    model = load_and_cache_model("rbc_linearized.gcn")
    expected = model.linearize_model(verbose=False)
    A, B, _C, _D = expected

    with caplog.at_level("WARNING"):
        outputs = _maybe_linearize_model(model, A, B, None, None, verbose=verbose)

    for actual, reference in zip(outputs, expected, strict=True):
        assert_allclose(actual, reference)
    assert ("Passing an incomplete subset of A, B, C, and D (you passed 2)" in caplog.text) == verbose


def test_compute_stationary_covariance_warns_on_partial_specification(caplog):
    model = load_and_cache_model("rbc_linearized.gcn")
    T, _R = model.solve_model(solver="gensys", verbose=False)

    stationary_covariance_matrix(model, T, shock_std=0.1, verbose=False)
    assert caplog.messages[-1].startswith("Passing only one of T or R will still trigger")


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "open_rbc.gcn",
        pytest.param("full_nk.gcn", marks=pytest.mark.include_nk),
        "rbc_linearized.gcn",
    ],
)
def test_compute_stationary_covariance(caplog, gcn_file):
    model = load_and_cache_model(gcn_file)
    T, R = model.solve_model(solver="gensys", verbose=False)
    n_variables, _n_shocks = R.shape

    Sigma = stationary_covariance_matrix(model, T, R, shock_std=0.1, return_df=False)
    assert len(caplog.messages) == 0
    assert Sigma.shape == (n_variables, n_variables)

    assert_allclose(Sigma, Sigma.T, atol=1e-8)
    assert all(x > 0 for x in np.diagonal(Sigma))

    # Sigma is positive semidefinite when clipping its negative eigenvalues to zero leaves it unchanged.
    eigvals, eigvecs = np.linalg.eig(Sigma)
    eigvals = np.where(eigvals < 0, 0, eigvals)
    Sigma_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    assert_allclose(Sigma, Sigma_psd, atol=1e-8)


@pytest.mark.parametrize(
    "gcn_file, state_name, rho_name",
    [
        ("one_block_1_ss.gcn", "A", "rho"),
        ("open_rbc.gcn", "A", "rho_A"),
        ("rbc_linearized.gcn", "A", "rho_A"),
        pytest.param("full_nk.gcn", "shock_technology", "rho_technology", marks=pytest.mark.include_nk),
        pytest.param("full_nk.gcn", "shock_preference", "rho_preference", marks=pytest.mark.include_nk),
    ],
)
def test_autocorrelation_of_ar1_state_decays_at_rho(gcn_file, state_name, rho_name, rng):
    model = load_and_cache_model(gcn_file)
    state_idx = model.variables.index(model.get(state_name))
    rho_value = rng.beta(10, 1)

    autocorr = autocorrelation_matrix(
        model,
        shock_std=0.1,
        solver="gensys",
        verbose=False,
        return_xr=False,
        **{rho_name: rho_value},
    )

    assert_allclose(autocorr[:, state_idx, state_idx], rho_value ** np.arange(10), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "open_rbc.gcn",
        pytest.param("full_nk.gcn", marks=pytest.mark.include_nk),
        "rbc_linearized.gcn",
    ],
)
def test_autocovariance_matrix_lag_zero_is_stationary_covariance(gcn_file):
    model = load_and_cache_model(gcn_file)
    shock_std_dict = {shock.base_name: 0.1 for shock in model.shocks}

    autocov = autocovariance_matrix(
        model, shock_std_dict=shock_std_dict, solver="gensys", verbose=False, return_xr=False
    )
    Sigma = stationary_covariance_matrix(
        model, shock_std_dict=shock_std_dict, solver="gensys", verbose=False, return_df=False
    )

    assert_allclose(autocov[0], Sigma, atol=1e-8, rtol=1e-8)


def _shock_covariance_arguments(argument, n_shocks, model):
    shock_std = 0.1 if argument == "shock_std" else None
    shock_std_dict = {shock.base_name: 0.1 for shock in model.shocks} if argument == "shock_std_dict" else None
    shock_cov_matrix = np.eye(n_shocks) * 0.1**2 if argument == "shock_cov_matrix" else None

    return shock_std, shock_std_dict, shock_cov_matrix


@pytest.fixture
def irf_inputs():
    model = load_and_cache_model("one_block_1_ss_2shock.gcn")
    T, R = model.solve_model(solver="gensys", verbose=False)
    return model, T, R


SHOCK_SIZE_CASES = [
    (0.1, ["epsilon_A", "epsilon_B"]),
    (np.array([0.1, 0.1]), ["epsilon_A", "epsilon_B"]),
    ({"epsilon_A": 0.1, "epsilon_B": 0.1}, ["epsilon_A", "epsilon_B"]),
    ({"epsilon_B": 0.1}, ["epsilon_B"]),
]
SHOCK_SIZE_IDS = ["single_float", "array", "dict", "partial_dict"]


class TestIRF:
    @pytest.mark.parametrize("shock_size, expected_shocks", SHOCK_SIZE_CASES, ids=SHOCK_SIZE_IDS)
    def test_irf_from_shock_size_with_individual_shocks(self, irf_inputs, shock_size, expected_shocks):
        model, T, R = irf_inputs
        n_variables, _n_shocks = R.shape

        irf = impulse_response_function(
            model, T, R, simulation_length=1000, shock_size=shock_size, return_individual_shocks=True
        )

        assert dict(irf.sizes) == {"shock": len(expected_shocks), "time": 1000, "variable": n_variables}
        assert list(irf.coords["shock"].values) == expected_shocks

        # After 1000 periods the responses have died out.
        assert np.all(np.abs(irf.isel(time=-1).values) < 1e-3)

        for first, second in itertools.combinations(expected_shocks, 2):
            assert not np.allclose(irf.sel(shock=first).values, irf.sel(shock=second).values)

    @pytest.mark.parametrize("shock_size, _expected_shocks", SHOCK_SIZE_CASES, ids=SHOCK_SIZE_IDS)
    def test_irf_from_shock_size_with_joint_shocks(self, irf_inputs, shock_size, _expected_shocks):
        model, T, R = irf_inputs
        n_variables, _n_shocks = R.shape

        irf = impulse_response_function(
            model, T, R, simulation_length=1000, shock_size=shock_size, return_individual_shocks=False
        )

        assert dict(irf.sizes) == {"time": 1000, "variable": n_variables}
        assert np.all(np.abs(irf.isel(time=-1).values) < 1e-3)

    @pytest.mark.parametrize("n_shocked", [1, 2], ids=["single_shock", "two_shocks"])
    def test_irf_from_trajectory_with_individual_shocks(self, irf_inputs, n_shocked):
        model, T, R = irf_inputs
        n_variables, n_shocks = R.shape

        shock_trajectory = np.zeros((1000, n_shocks))
        shock_trajectory[0, :n_shocked] = 0.1

        irf = impulse_response_function(
            model, T, R, simulation_length=1000, shock_trajectory=shock_trajectory, return_individual_shocks=True
        )

        assert dict(irf.sizes) == {"shock": n_shocks, "time": 1000, "variable": n_variables}
        assert np.all(np.abs(irf.isel(time=-1).values) < 1e-3)
        assert not np.allclose(irf.sel(shock="epsilon_A").values, irf.sel(shock="epsilon_B").values)

    @pytest.mark.parametrize("n_shocked", [1, 2], ids=["single_shock", "two_shocks"])
    def test_irf_from_trajectory_with_joint_shocks(self, irf_inputs, n_shocked):
        model, T, R = irf_inputs
        n_variables, n_shocks = R.shape

        shock_trajectory = np.zeros((1000, n_shocks))
        shock_trajectory[0, :n_shocked] = 0.1

        irf = impulse_response_function(
            model, T, R, simulation_length=1000, shock_trajectory=shock_trajectory, return_individual_shocks=False
        )

        assert dict(irf.sizes) == {"time": 1000, "variable": n_variables}
        assert np.all(np.abs(irf.isel(time=-1).values) < 1e-3)

    def test_size_dict_limits_shock_axis(self, irf_inputs):
        model, T, R = irf_inputs

        da = impulse_response_function(
            model, T=T, R=R, shock_size={"epsilon_A": 1.0, "epsilon_B": 0.5}, simulation_length=5
        )
        assert "shock" in da.dims
        assert list(da.coords["shock"].values) == ["epsilon_A", "epsilon_B"]

    def test_size_dict_empty_raises(self, irf_inputs):
        model, T, R = irf_inputs
        with pytest.raises(ValueError):
            impulse_response_function(model, T=T, R=R, shock_size={})

    def test_size_dict_combined_has_no_shock_axis(self, irf_inputs):
        model, T, R = irf_inputs
        da = impulse_response_function(
            model, T=T, R=R, shock_size={"epsilon_A": 1.0}, simulation_length=5, return_individual_shocks=False
        )
        assert "shock" not in da.dims


@pytest.mark.parametrize(
    "gcn_file",
    [
        "one_block_1_ss.gcn",
        "open_rbc.gcn",
        pytest.param("full_nk.gcn", marks=pytest.mark.include_nk),
    ],
)
@pytest.mark.parametrize("argument", ["shock_std", "shock_std_dict", "shock_cov_matrix"])
def test_simulate(gcn_file, argument):
    model = load_and_cache_model(gcn_file)
    T, R = model.solve_model(solver="gensys", verbose=False)
    n_variables, n_shocks = R.shape

    n_simulations = 3000
    simulation_length = 2000

    shock_std, shock_std_dict, shock_cov_matrix = _shock_covariance_arguments(argument, n_shocks, model)

    data = simulate(
        model,
        T,
        R,
        simulation_length=simulation_length,
        n_simulations=n_simulations,
        shock_std=shock_std,
        shock_std_dict=shock_std_dict,
        shock_cov_matrix=shock_cov_matrix,
        random_seed=1234,
    )

    assert data.shape == (n_simulations, simulation_length, n_variables)

    # Across many trajectories the terminal-period sample covariance tracks the stationary covariance.
    Sigma = stationary_covariance_matrix(model, T, R, shock_std=0.1, return_df=False)
    sigma = np.cov(data.isel(time=-1).values.T)

    corr = np.corrcoef(Sigma.ravel(), sigma.ravel())[0, 1]
    assert corr > 0.99

    assert_allclose(np.diag(Sigma), np.diag(sigma), rtol=0.1)


def test_objective_with_complex_discount_factor():
    model = load_and_cache_model("rbc_firm_capital.gcn")

    ss_res = model.steady_state(verbose=False, how="minimize", optimizer_kwargs={"method": "Newton-CG"})
    assert ss_res.success

    bk_success = check_bk_condition(
        *model.linearize_model(steady_state=ss_res),
        return_value="bool",
        verbose=False,
    )
    assert bk_success

    model_2 = load_and_cache_model("rbc_firm_capital_comparison.gcn")
    ss_res_2 = model_2.steady_state(verbose=False)
    assert ss_res_2.success

    for name in ["Y_ss", "K_ss", "L_ss", "I_ss"]:
        assert_allclose(ss_res[name], ss_res_2[name], rtol=1e-8, atol=1e-8)
