import os

import numpy as np
import pytest
import sympy as sp

from numpy.testing import assert_allclose

from gEconpy.model.perturbation import (
    linearize_model,
    make_all_variable_time_combinations,
    override_dummy_wrapper,
    solve_policy_function_with_cycle_reduction,
    solve_policy_function_with_gensys,
)
from gEconpy.shared.utilities import eq_to_ss
from tests.utilities.shared_fixtures import load_and_cache_model


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
def test_variables_to_all_times(load_and_cache_model, backend):
    mod = load_and_cache_model("One_Block_Simple_1.gcn", backend)
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
    ["One_Block_Simple_1.gcn", "Two_Block_RBC_1.gcn", "Full_New_Keynesian.gcn"],
)
@pytest.mark.parametrize("backend", ["numpy", "numba", "pytensor"])
def test_log_linearize_model(load_and_cache_model, gcn_file, backend):
    mod = load_and_cache_model(gcn_file, backend)
    (A, B, C, D), not_loglin_variable = linearize_model(
        mod.variables, mod.equations, mod.shocks
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
        ("One_Block_Simple_1_w_Steady_State.gcn", ["K", "A"]),
        ("Open_RBC.gcn", ["A", "K", "IIP"]),
        (
            "Full_New_Keynesian.gcn",
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
def test_solve_policy_function(
    load_and_cache_model, gcn_file, state_variables, backend
):
    mod = load_and_cache_model(gcn_file, backend)
    steady_state_dict, success = mod.steady_state()
    A, B, C, D = mod.linearize_model(order=1, steady_state_dict=steady_state_dict)

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
