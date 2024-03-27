import os

import numpy as np
import pytest
import sympy as sp

from gEconpy.model.build import model_from_gcn
from gEconpy.model.perturbation.perturbation import (
    linearize_model,
    make_all_variable_time_combinations,
)
from gEconpy.shared.utilities import eq_to_ss


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


def test_variables_to_all_times():
    mod = model_from_gcn("tests/Test GCNs/One_Block_Simple_1.gcn", verbose=False)
    variables = mod.variables
    lags, now, leads = make_all_variable_time_combinations(variables)

    assert set(variables) == set(now)
    assert all([len(vars) == len(variables) for vars in [lags, now, leads]])
    for i, var_group in enumerate([lags, now, leads]):
        t = i - 1
        assert all([var.time_index == t for var in var_group])
        assert all([var.set_t(0) in mod.variables for var in var_group])


@pytest.mark.parametrize(
    "gcn_file", ["One_Block_Simple_1.gcn", "Two_Block_RBC_1.gcn", "Full_New_Keyensian.gcn"]
)
def test_log_linearize_model(gcn_file):
    mod = model_from_gcn(os.path.join("tests/Test GCNs", gcn_file), verbose=False)
    A, B, C, D = linearize_model(mod.variables, mod.equations, mod.shocks)
    lags, now, leads = make_all_variable_time_combinations(mod.variables)

    ss_vars = [x.to_ss() for x in mod.variables]
    ss_shocks = [x.to_ss() for x in mod.shocks]
    parameters = list(mod.parameters().to_sympy().keys())

    # not_loglin_variables = [
    #     sp.Symbol(f"{x.base_name}_not_login", postive=True) for x in mod.variables
    # ]

    not_loglin_variable = sp.IndexedBase("not_loglin_variable")

    sub_dict = {x.name: 0.8 for x in ss_vars}
    shock_dict = {x.to_ss().name: 0.0 for x in mod.shocks}

    A2, B2, C2, D2 = linearize_method_2(now, mod.equations, mod.shocks)
    A22, B22, C22, D22 = linearize_method_2(
        now, mod.equations, mod.shocks, not_loglin_variables=[x.base_name for x in mod.variables]
    )

    for i, (M1, M2) in enumerate(zip([A, B, C, D], [(A2, A22), (B2, B22), (C2, C22), (D2, D22)])):
        f1 = sp.lambdify(ss_vars + ss_shocks + parameters + [not_loglin_variable], M1)
        f2 = sp.lambdify(ss_vars + ss_shocks + parameters, list(M2))

        for loglin_value in [0, 1]:
            x = f1(
                **mod.parameters(),
                **sub_dict,
                **shock_dict,
                **dict.fromkeys([x.name for x in not_loglin_variable], loglin_value),
            )
            x2 = f2(**mod.parameters(), **sub_dict, **shock_dict)

            np.testing.assert_allclose(x, x2[loglin_value])


@pytest.mark.parametrize(
    "gcn_file", ["One_Block_Simple_1.gcn", "Two_Block_RBC_1.gcn", "Full_New_Keyensian.gcn"]
)
def test_solve_with_gensys(gcn_file):
    mod = model_from_gcn(os.path.join("tests/Test GCNs", gcn_file), verbose=False)
