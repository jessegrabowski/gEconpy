from importlib.util import find_spec

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import sympy as sp

from numpy.testing import assert_allclose
from pytensor.gradient import DisconnectedType, verify_grad

from gEconpy.model.perturbation import (
    linearize_model,
    make_all_variable_time_combinations,
    override_dummy_wrapper,
)
from gEconpy.solvers.cycle_reduction import (
    cycle_reduction_pt,
    scan_cycle_reduction,
    solve_policy_function_with_cycle_reduction,
)
from gEconpy.solvers.gensys import gensys_pt, solve_policy_function_with_gensys
from gEconpy.utilities import eq_to_ss
from tests.utilities.shared_fixtures import load_and_cache_model


JAX_INSTALLED = find_spec("jax") is not None


def linearize_method_2(variables, equations, shocks, not_loglin_variables=None):
    if not_loglin_variables is None:
        not_loglin_variables = []
    not_loglin_variables += [x.base_name for x in shocks]

    Fs = []
    lags, now, leads = make_all_variable_time_combinations(variables)

    for var_group in [lags, now, leads, shocks]:
        F = []
        for eq in equations:
            F_row = []
            for var in var_group:
                dydx = sp.powsimp(eq_to_ss(eq.diff(var)))
                dydx *= 1.0 if var.base_name in not_loglin_variables else var.to_ss()
                F_row.append(dydx)
            F.append(F_row)
        F = sp.Matrix(F)
        Fs.append(F)
    return Fs


@pytest.mark.parametrize("backend", ["numpy", "numba", "pytensor"])
def test_variables_to_all_times(backend):
    mod = load_and_cache_model(
        "one_block_1.gcn", backend=backend, use_jax=JAX_INSTALLED
    )
    variables = mod.variables
    lags, now, leads = make_all_variable_time_combinations(variables)

    assert set(variables) == set(now)
    assert all([len(vars) == len(variables) for vars in [lags, now, leads]])
    for i, var_group in enumerate([lags, now, leads]):
        t = i - 1
        assert all([var.time_index == t for var in var_group])
        assert all([var.set_t(0) in mod.variables for var in var_group])


@pytest.mark.parametrize(
    "gcn_file",
    ["one_block_1.gcn", "rbc_2_block.gcn", "full_nk.gcn"],
)
@pytest.mark.parametrize("backend", ["numpy", "numba", "pytensor"])
def test_log_linearize_model(gcn_file, backend):
    mod = load_and_cache_model(gcn_file, backend=backend, use_jax=JAX_INSTALLED)
    (A, B, C, D), not_loglin_variable = linearize_model(
        mod.variables,
        mod.equations,
        mod.shocks,
    )
    lags, now, leads = make_all_variable_time_combinations(mod.variables)

    ss_vars = [x.to_ss() for x in mod.variables]
    ss_shocks = [x.to_ss() for x in mod.shocks]
    parameters = list(mod.parameters().to_sympy().keys())

    sub_dict = {x.name: 0.8 for x in ss_vars}
    shock_dict = {x.to_ss().name: 0.0 for x in mod.shocks}

    A2, B2, C2, D2 = linearize_method_2(now, mod.equations, mod.shocks)
    A22, B22, C22, D22 = linearize_method_2(
        now,
        mod.equations,
        mod.shocks,
        not_loglin_variables=[x.base_name for x in mod.variables],
    )

    for i, (M1, M2) in enumerate(
        zip([A, B, C, D], [(A2, A22), (B2, B22), (C2, C22), (D2, D22)])
    ):
        f1 = sp.lambdify(ss_vars + ss_shocks + parameters + [not_loglin_variable], M1)
        f1 = override_dummy_wrapper(f1, "not_loglin_variable")
        f2 = sp.lambdify(ss_vars + ss_shocks + parameters, list(M2))

        for loglin_value in [0, 1]:
            x = f1(
                **mod.parameters(),
                **sub_dict,
                **shock_dict,
                not_loglin_variable=np.full(len(ss_vars), loglin_value),
            )
            x2 = f2(**mod.parameters(), **sub_dict, **shock_dict)

            np.testing.assert_allclose(x, x2[loglin_value])


@pytest.mark.parametrize(
    "gcn_file, state_variables",
    [
        ("one_block_1_ss.gcn", ["K", "A"]),
        ("open_rbc.gcn", ["A", "K", "IIP"]),
        (
            "full_nk.gcn",
            [
                "K",
                "C",
                "I",
                "Y",
                "w",
                "pi_star",
                "shock_technology",
                "shock_preference",
                "pi_obj",
                "r_G",
            ],
        ),
    ],
)
@pytest.mark.parametrize("backend", ["numpy", "numba", "pytensor"])
def test_solve_policy_function(gcn_file, state_variables, backend):
    mod = load_and_cache_model(gcn_file, backend=backend, use_jax=JAX_INSTALLED)
    steady_state_dict = mod.steady_state()
    A, B, C, D = mod.linearize_model(
        order=1,
        steady_state=steady_state_dict,
        verbose=False,
        steady_state_kwargs={"verbose": False, "progressbar": False},
    )

    A, B, C, D = [np.ascontiguousarray(x, dtype="float64") for x in [A, B, C, D]]

    gensys_results = solve_policy_function_with_gensys(A, B, C, D, 1e-8)
    G_1, constant, impact, f_mat, f_wt, y_wt, gev, eu, loose = gensys_results

    state_idxs = [
        i for i, var in enumerate(mod.variables) if var.base_name in state_variables
    ]
    jumper_idxs = [
        i for i, var in enumerate(mod.variables) if var.base_name not in state_variables
    ]

    assert not np.allclose(G_1[:, state_idxs], 0.0)
    assert_allclose(G_1[:, jumper_idxs], 0.0, atol=1e-8, rtol=1e-8)

    n = len(mod.variables)
    T_gensys = G_1[:n, :][:, :n]
    R_gensys = impact[:n, :]

    (
        T,
        R,
        result,
        log_norm,
    ) = solve_policy_function_with_cycle_reduction(A, B, C, D, 100_000, 1e-16, False)

    assert not np.allclose(T[:, state_idxs], 0.0)
    assert_allclose(T[:, jumper_idxs], 0.0, atol=1e-8, rtol=1e-8)

    assert_allclose(T_gensys, T, atol=1e-8, rtol=1e-8)
    assert_allclose(R_gensys, R, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    "op",
    [cycle_reduction_pt, scan_cycle_reduction],
    ids=["cycle_reduction", "scan_cycle_reduction"],
)
def test_cycle_reduction_gradients(op):
    mod = load_and_cache_model("full_nk.gcn", backend="numpy", use_jax=JAX_INSTALLED)
    A, B, C, D = mod.linearize_model(
        verbose=False, steady_state_kwargs={"verbose": False, "progressbar": False}
    )
    A, B, C, D = [np.ascontiguousarray(x, dtype="float64") for x in [A, B, C, D]]

    A_pt, B_pt, C_pt, D_pt = (
        pt.tensor(name=name, shape=x.shape)
        for name, x in zip(list("ABCD"), [A, B, C, D])
    )

    T, R, *_ = op(A_pt, B_pt, C_pt, D_pt)
    T_grad = pt.grad(T.sum(), [A_pt, B_pt, C_pt])

    f = pytensor.function(
        [A_pt, B_pt, C_pt, D_pt],
        [T, R, *T_grad],
        on_unused_input="raise",
        mode="JAX" if JAX_INSTALLED and op is scan_cycle_reduction else "FAST_RUN",
    )

    T_np, R_np, A_bar, B_bar, C_bar = f(A, B, C, D)

    resid = A + B @ T_np + C @ T_np @ T_np
    assert_allclose(resid, 0.0, atol=1e-8, rtol=1e-8)

    def cycle_func(A, B, C, D):
        T, R, *_ = op(A, B, C, D)
        return T.sum()

    verify_grad(
        cycle_func, pt=[A, B, C, D.astype("float64")], rng=np.random.default_rng()
    )


def test_pytensor_gensys():
    mod = load_and_cache_model("full_nk.gcn", backend="numpy", use_jax=JAX_INSTALLED)
    A, B, C, D = mod.linearize_model(
        verbose=False, steady_state_kwargs={"verbose": False, "progressbar": False}
    )

    A_pt, B_pt, C_pt, D_pt = (pt.dmatrix(name) for name in list("ABCD"))
    T1, R1 = cycle_reduction_pt(A_pt, B_pt, C_pt, D_pt)
    T1_grad = pt.grad(T1.sum(), [A_pt, B_pt, C_pt])

    T2, R2, success = gensys_pt(A_pt, B_pt, C_pt, D_pt, 1e-8)
    T2_grad = pt.grad(T2.sum(), [A_pt, B_pt, C_pt])

    def gensys_func(A, B, C, D):
        T, R, _ = gensys_pt(A, B, C, D)
        return T.sum()

    verify_grad(
        gensys_func, pt=[A, B, C, D.astype("float64")], rng=np.random.default_rng()
    )

    f = pytensor.function(
        [A_pt, B_pt, C_pt, D_pt],
        [T1, T2, R1, R2, *T1_grad, *T2_grad],
        on_unused_input="raise",
        mode="FAST_RUN",
    )

    T1_np, T2_np, R1_np, R2_np, A_bar_1, B_bar_1, C_bar_1, A_bar_2, B_bar_2, C_bar_2 = (
        f(A, B, C, D)
    )

    assert_allclose(A_bar_1, A_bar_2, atol=1e-8, rtol=1e-8)
    assert_allclose(B_bar_1, B_bar_2, atol=1e-8, rtol=1e-8)
    assert_allclose(C_bar_1, C_bar_2, atol=1e-8, rtol=1e-8)
