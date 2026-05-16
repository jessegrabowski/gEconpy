import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import sympy as sp

from numpy.testing import assert_allclose
from pytensor.gradient import verify_grad
from pytensor.graph.traversal import explicit_graph_inputs

from gEconpy.model.perturbation import linearize_model
from gEconpy.model.timing import make_all_variable_time_combinations
from gEconpy.pytensorf.compile import compile_pytensor_function
from gEconpy.solvers.cycle_reduction import (
    cycle_reduction_pt,
    scan_cycle_reduction,
    solve_policy_function_with_cycle_reduction,
)
from gEconpy.solvers.gensys import gensys_pt, solve_policy_function_with_gensys
from gEconpy.utilities import eq_to_ss
from tests._resources.cache_compiled_models import load_and_cache_model


def _sympy_jacobians(variables, equations, shocks, not_loglin_variables=None):
    """Compute Jacobian matrices via direct sympy differentiation (ground-truth reference).

    For each variable group (lags, current, leads, shocks), computes dF/dvar at steady state. For log-linearized
    variables, the derivative is multiplied by the steady-state value (the T-matrix column).
    """
    if not_loglin_variables is None:
        not_loglin_variables = []
    not_loglin_variables += [x.base_name for x in shocks]
    # Variables declared negative cannot be log-linearized
    not_loglin_variables += [v.base_name for v in variables if v.assumptions0.get("negative", False)]

    lags, now, leads = make_all_variable_time_combinations(variables)
    matrices = []
    for var_group in [lags, now, leads, shocks]:
        rows = []
        for eq in equations:
            row = []
            for var in var_group:
                deriv = sp.powsimp(eq_to_ss(eq.diff(var)))
                if var.base_name not in not_loglin_variables:
                    deriv *= var.to_ss()
                row.append(deriv)
            rows.append(row)
        matrices.append(sp.Matrix(rows))
    return matrices


def _compile_and_eval(mod, jacobians, ss_nodes):
    """Compile a list of pytensor Jacobian graphs and evaluate at dummy SS values."""
    ss_names = {n.name for n in ss_nodes}
    param_inputs = [v for v in explicit_graph_inputs(jacobians) if v.name is not None and v.name not in ss_names]

    f = compile_pytensor_function(ss_nodes + param_inputs, jacobians, on_unused_input="ignore")

    ss_vals = [0.8] * len(ss_nodes)
    param_vals = [mod.parameters().get(n.name, 0.0) for n in param_inputs]
    return f(*ss_vals, *param_vals)


class TestMakeAllVariableTimeCombinations:
    def test_produces_lags_now_leads(self):
        mod = load_and_cache_model("one_block_1.gcn")
        lags, now, leads = make_all_variable_time_combinations(mod.variables)

        assert set(mod.variables) == set(now)
        assert len(lags) == len(now) == len(leads) == len(mod.variables)
        for offset, group in [(-1, lags), (0, now), (1, leads)]:
            assert all(v.time_index == offset for v in group)
            assert all(v.set_t(0) in mod.variables for v in group)


class TestLinearizeModel:
    """Verify the pytensor-based linearize_model against direct sympy differentiation."""

    @pytest.mark.parametrize(
        "gcn_file",
        [
            "one_block_1.gcn",
            "rbc_2_block.gcn",
            "open_rbc.gcn",
            pytest.param("full_nk.gcn", marks=pytest.mark.include_nk),
        ],
    )
    def test_loglin_matches_sympy(self, gcn_file):
        mod = load_and_cache_model(gcn_file)

        jacobians, ss_inputs = linearize_model(
            mod.variables,
            mod.equations,
            mod.shocks,
            cache={},
            loglin_variables=mod.variables,
        )
        actual = _compile_and_eval(mod, jacobians, ss_inputs)

        expected_mats = _sympy_jacobians(mod.variables, mod.equations, mod.shocks)
        subs = {
            **mod.parameters().to_sympy(),
            **{x.to_ss(): 0.8 for x in mod.variables},
            **{x.to_ss(): 0.0 for x in mod.shocks},
        }

        for name, actual_mat, sym_mat in zip("ABCD", actual, expected_mats, strict=False):
            expected = np.array(sym_mat.subs(subs)).astype(float)
            assert_allclose(actual_mat, expected, atol=1e-10, err_msg=f"loglin {name} mismatch for {gcn_file}")

    @pytest.mark.parametrize(
        "gcn_file",
        [
            "one_block_1.gcn",
            "rbc_2_block.gcn",
            "open_rbc.gcn",
            pytest.param("full_nk.gcn", marks=pytest.mark.include_nk),
        ],
    )
    def test_no_loglin_matches_sympy(self, gcn_file):
        mod = load_and_cache_model(gcn_file)

        jacobians, ss_inputs = linearize_model(
            mod.variables,
            mod.equations,
            mod.shocks,
            cache={},
            loglin_variables=[],
        )
        actual = _compile_and_eval(mod, jacobians, ss_inputs)

        all_excluded = [x.base_name for x in mod.variables]
        expected_mats = _sympy_jacobians(mod.variables, mod.equations, mod.shocks, not_loglin_variables=all_excluded)
        subs = {
            **mod.parameters().to_sympy(),
            **{x.to_ss(): 0.8 for x in mod.variables},
            **{x.to_ss(): 0.0 for x in mod.shocks},
        }

        for name, actual_mat, sym_mat in zip("ABCD", actual, expected_mats, strict=False):
            expected = np.array(sym_mat.subs(subs)).astype(float)
            assert_allclose(actual_mat, expected, atol=1e-10, err_msg=f"no-loglin {name} mismatch for {gcn_file}")


class TestSolvePolicyFunction:
    """Verify that gensys and cycle reduction produce the same policy function, with correct state/jumper structure."""

    @pytest.mark.parametrize(
        "gcn_file, state_variables",
        [
            ("one_block_1_ss.gcn", ["K", "A"]),
            ("open_rbc.gcn", ["A", "K", "IIP"]),
            pytest.param(
                "full_nk.gcn",
                ["K", "C", "I", "Y", "w", "pi_star", "shock_technology", "shock_preference", "pi_obj", "r_G"],
                marks=pytest.mark.include_nk,
            ),
        ],
    )
    def test_gensys_and_cycle_reduction_agree(self, gcn_file, state_variables):
        mod = load_and_cache_model(gcn_file)
        ss = mod.steady_state()
        A, B, C, D = [
            np.ascontiguousarray(x, dtype="float64")
            for x in mod.linearize_model(
                order=1,
                steady_state=ss,
                verbose=False,
                steady_state_kwargs={"verbose": False, "progressbar": False},
            )
        ]

        state_idxs = [i for i, v in enumerate(mod.variables) if v.base_name in state_variables]
        jumper_idxs = [i for i, v in enumerate(mod.variables) if v.base_name not in state_variables]
        n = len(mod.variables)

        G_1, _, impact, *_ = solve_policy_function_with_gensys(A, B, C, D, 1e-8)
        T_gensys = G_1[:n, :n]
        R_gensys = impact[:n, :]

        T_cr, R_cr, *_ = solve_policy_function_with_cycle_reduction(A, B, C, D, 100_000, 1e-16, False)

        for T in [T_gensys, T_cr]:
            assert not np.allclose(T[:, state_idxs], 0.0), "State columns should be non-zero"
            assert_allclose(T[:, jumper_idxs], 0.0, atol=1e-8)

        assert_allclose(T_gensys, T_cr, atol=1e-8, rtol=1e-8)
        assert_allclose(R_gensys, R_cr, atol=1e-8, rtol=1e-8)


class TestCycleReductionGradients:
    @pytest.mark.parametrize(
        "op", [cycle_reduction_pt, scan_cycle_reduction], ids=["cycle_reduction", "scan_cycle_reduction"]
    )
    @pytest.mark.include_nk
    def test_gradients_verify(self, op):
        mod = load_and_cache_model("full_nk.gcn")
        A, B, C, D = [
            np.ascontiguousarray(x, dtype="float64")
            for x in mod.linearize_model(
                verbose=False,
                steady_state_kwargs={"verbose": False, "progressbar": False},
            )
        ]

        A_pt, B_pt, C_pt, D_pt = (
            pt.tensor(name=name, shape=x.shape) for name, x in zip("ABCD", [A, B, C, D], strict=False)
        )

        T, R, *_ = op(A_pt, B_pt, C_pt, D_pt)
        T_grad = pt.grad(T.sum(), [A_pt, B_pt, C_pt])

        f = pytensor.function([A_pt, B_pt, C_pt, D_pt], [T, R, *T_grad], on_unused_input="raise", mode="FAST_RUN")
        T_np, *_ = f(A, B, C, D)

        resid = A + B @ T_np + C @ T_np @ T_np
        assert_allclose(resid, 0.0, atol=1e-8, rtol=1e-8)

        verify_grad(lambda *args: op(*args)[0].sum(), pt=[A, B, C, D.astype("float64")], rng=np.random.default_rng())


class TestGensysPytensor:
    @pytest.mark.include_nk
    def test_gensys_and_cycle_reduction_gradients_agree(self):
        mod = load_and_cache_model("full_nk.gcn")
        A, B, C, D = [
            np.ascontiguousarray(x, dtype="float64")
            for x in mod.linearize_model(
                verbose=False,
                steady_state_kwargs={"verbose": False, "progressbar": False},
            )
        ]

        A_pt, B_pt, C_pt, D_pt = (pt.dmatrix(name) for name in "ABCD")

        T_cr, R_cr = cycle_reduction_pt(A_pt, B_pt, C_pt, D_pt)
        cr_grads = pt.grad(T_cr.sum(), [A_pt, B_pt, C_pt])

        T_gs, R_gs, _ = gensys_pt(A_pt, B_pt, C_pt, D_pt, 1e-8)
        gs_grads = pt.grad(T_gs.sum(), [A_pt, B_pt, C_pt])

        f = pytensor.function(
            [A_pt, B_pt, C_pt, D_pt],
            [T_cr, T_gs, R_cr, R_gs, *cr_grads, *gs_grads],
            on_unused_input="raise",
            mode="FAST_RUN",
        )
        _, _, _, _, A_bar_cr, B_bar_cr, C_bar_cr, A_bar_gs, B_bar_gs, C_bar_gs = f(A, B, C, D)

        assert_allclose(A_bar_cr, A_bar_gs, atol=1e-8, rtol=1e-8)
        assert_allclose(B_bar_cr, B_bar_gs, atol=1e-8, rtol=1e-8)
        assert_allclose(C_bar_cr, C_bar_gs, atol=1e-8, rtol=1e-8)

        verify_grad(
            lambda *args: gensys_pt(*args)[0].sum(),
            pt=[A, B, C, D.astype("float64")],
            rng=np.random.default_rng(),
        )
