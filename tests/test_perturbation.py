import os

import pytest
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.build import model_from_gcn
from gEconpy.model.perturbation.perturbation import (
    log_linearize_model,
    make_all_variable_time_combinations,
)
from gEconpy.shared.utilities import eq_to_ss


def pert_method_2(variables, equations, shocks, not_loglin_variables=None):
    if not_loglin_variables is None:
        not_loglin_variables = []

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


@pytest.mark.parametrize(
    "gcn_file", ["One_Block_Simple_1.gcn", "Two_Block_RBC_1.gcn", "Full_New_Keyensian.gcn"]
)
def test_log_linearize_model(gcn_file):
    mod = model_from_gcn(os.path.join("tests/Test GCNs", gcn_file), verbose=False)
    var_names = [var.base_name for var in mod.variables]
    A, B, C, D = log_linearize_model(mod.variables, mod.equations, mod.shocks)

    all_variables = set(
        [x for eq in mod.equations for x in eq.atoms() if isinstance(x, TimeAwareSymbol)]
    )
    all_variables -= set(mod.shocks)
    lags = [var for var in all_variables if var.time_index == -1]
    now = [var for var in all_variables if var.time_index == 0]
    leads = [var for var in all_variables if var.time_index == 1]

    not_loglin_variables = [
        sp.Symbol(f"{x.base_name}_not_login", postive=True) for x in mod.variables
    ]
    sub_dict = {x: 0.0 for x in not_loglin_variables}

    for var_group, pert_matrix in zip([lags, now, leads], [A, B, C]):
        for var in var_group:
            var_idx = var_names.index(var.base_name)
            has_var = [i for i, eq in enumerate(mod.equations) if var in eq.atoms()]
            for eq_idx in has_var:
                cell = pert_matrix[eq_idx, var_idx].subs(sub_dict)
                assert cell == eq_to_ss(mod.equations[eq_idx].diff(var) * var.to_ss())
                assert all(
                    [x.time_index == "ss" for x in cell.atoms() if isinstance(x, TimeAwareSymbol)]
                )

    A2, B2, C2, D2 = pert_method_2(mod.variables, mod.equations, mod.shocks)

    for i, (pert_matrix, pert_matrix2) in enumerate(zip([A, B, C], [A2, B2, C2])):
        assert pert_matrix.subs(sub_dict) == pert_matrix2

    assert False
