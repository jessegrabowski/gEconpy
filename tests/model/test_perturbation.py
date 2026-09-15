import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import sympy as sp

from numpy.testing import assert_allclose
from pytensor.gradient import verify_grad
from pytensor.graph.traversal import explicit_graph_inputs

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.build import model_from_gcn
from gEconpy.model.perturbation import (
    check_bk_condition,
    check_bk_condition_pt,
    linearize_model,
    make_not_loglin_flags,
)
from gEconpy.model.timing import make_all_variable_time_combinations
from gEconpy.pytensorf.compile import compile_pytensor_function
from gEconpy.solvers.cycle_reduction import (
    cycle_reduction_numpy,
    cycle_reduction_pt,
    scan_cycle_reduction,
    solve_policy_function_with_cycle_reduction,
)
from gEconpy.solvers.gensys import gensys_pt, solve_policy_function_with_gensys
from gEconpy.utilities import eq_to_ss
from tests._resources.cache_compiled_models import load_and_cache_model


def _sympy_jacobians(variables, equations, shocks, not_loglin_variables=None):
    """
    Compute the Jacobians by direct sympy differentiation, as the reference for ``linearize_model``.

    For each variable group (lags, current, leads, shocks), differentiates every equation at the steady state. The
    derivative of a log-linearized variable is multiplied by its steady-state value.
    """
    if not_loglin_variables is None:
        not_loglin_variables = []
    not_loglin_variables += [x.base_name for x in shocks]
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


def _unpermute_abcd(matrices, eq_order, var_order):
    """Undo the row and column permutations applied by ``linearize_model``. D's shock columns carry no permutation."""
    inv_eq = np.argsort(eq_order)
    inv_var = np.argsort(var_order)
    A, B, C, D = matrices
    A = A[inv_eq][:, inv_var]
    B = B[inv_eq][:, inv_var]
    C = C[inv_eq][:, inv_var]
    D = D[inv_eq]
    return [A, B, C, D]


def _compile_and_eval(mod, jacobians, ss_nodes):
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
    @pytest.mark.parametrize("loglin", [True, False], ids=["loglin", "no_loglin"])
    def test_matches_sympy(self, gcn_file, loglin):
        mod = load_and_cache_model(gcn_file)

        jacobians, ss_inputs, eq_order, var_order = linearize_model(
            mod.variables,
            mod.equations,
            mod.shocks,
            cache={},
            loglin_variables=mod.variables if loglin else [],
        )
        actual = _compile_and_eval(mod, jacobians, ss_inputs)
        actual = _unpermute_abcd(actual, eq_order, var_order)

        not_loglin = [] if loglin else [x.base_name for x in mod.variables]
        expected_mats = _sympy_jacobians(mod.variables, mod.equations, mod.shocks, not_loglin_variables=not_loglin)
        subs = {
            **mod.parameters().to_sympy(),
            **{x.to_ss(): 0.8 for x in mod.variables},
            **{x.to_ss(): 0.0 for x in mod.shocks},
        }

        for name, actual_mat, sym_mat in zip("ABCD", actual, expected_mats, strict=True):
            expected = np.array(sym_mat.subs(subs)).astype(float)
            assert_allclose(actual_mat, expected, atol=1e-10, err_msg=f"{name} mismatch for {gcn_file}")

    def test_explicit_orderings_permute_rows_and_columns(self):
        """A caller-supplied permutation is applied verbatim and reported back, even when passed as a list."""
        mod = load_and_cache_model("rbc_2_block.gcn")
        n_eq, n_var = len(mod.equations), len(mod.variables)
        eq_order = list(range(n_eq))[::-1]
        var_order = list(range(n_var))[::-1]

        default_jacobians, default_ss, default_eq_order, default_var_order = linearize_model(
            mod.variables, mod.equations, mod.shocks, cache={}
        )
        jacobians, ss_inputs, eq_order_out, var_order_out = linearize_model(
            mod.variables, mod.equations, mod.shocks, cache={}, eq_order=eq_order, var_order=var_order
        )

        np.testing.assert_array_equal(eq_order_out, eq_order)
        np.testing.assert_array_equal(var_order_out, var_order)

        default_mats = _unpermute_abcd(
            _compile_and_eval(mod, default_jacobians, default_ss), default_eq_order, default_var_order
        )
        reversed_mats = _unpermute_abcd(_compile_and_eval(mod, jacobians, ss_inputs), eq_order, var_order)
        for name, default_mat, reversed_mat in zip("ABCD", default_mats, reversed_mats, strict=True):
            assert_allclose(reversed_mat, default_mat, atol=1e-12, err_msg=f"{name} differs under reversed ordering")


class TestMakeNotLoglinFlags:
    K, C, B = (TimeAwareSymbol(name, 0) for name in "KCB")
    alpha = sp.Symbol("alpha")
    steady_state = SymbolDictionary({K.to_ss(): 2.0, C.to_ss(): 0.0, B.to_ss(): -1.0, alpha: 0.3})

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, [0, 1, 1, 0]),
            ({"loglin_negative_ss": True}, [0, 1, 0, 0]),
            ({"not_loglin_variables": ["K", "alpha"]}, [1, 1, 1, 1]),
        ],
        ids=["zero_and_negative_ss", "allow_negative_ss", "user_exclusions"],
    )
    def test_flags(self, kwargs, expected):
        flags = make_not_loglin_flags(
            [self.K, self.C, self.B], [self.alpha], self.steady_state, verbose=False, **kwargs
        )
        np.testing.assert_array_equal(flags, expected)

    def test_log_linearize_false_flags_every_variable(self):
        flags = make_not_loglin_flags(
            [self.K, self.C, self.B], [self.alpha], self.steady_state, log_linearize=False, verbose=False
        )
        np.testing.assert_array_equal(flags, [1, 1, 1, 1])

    def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="unknown to the model: Z"):
            make_not_loglin_flags(
                [self.K, self.C, self.B], [self.alpha], self.steady_state, not_loglin_variables=["Z"], verbose=False
            )


class TestSolvePolicyFunction:
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

        G_1, _, impact, *_ = solve_policy_function_with_gensys(A, B, C, D, tol=1e-8)
        T_gensys = G_1[:n, :n]
        R_gensys = impact[:n, :]

        T_cr, R_cr, *_ = solve_policy_function_with_cycle_reduction(
            A, B, C, D, max_iter=100_000, tol=1e-16, verbose=False
        )

        for T in [T_gensys, T_cr]:
            assert not np.allclose(T[:, state_idxs], 0.0), "State columns should be non-zero"
            assert_allclose(T[:, jumper_idxs], 0.0, atol=1e-8)

        assert_allclose(T_gensys, T_cr, atol=1e-8, rtol=1e-8)
        assert_allclose(R_gensys, R_cr, atol=1e-8, rtol=1e-8)

    def test_cycle_reduction_reports_non_convergence(self):
        mod = model_from_gcn("tests/_resources/test_gcns/pert_fails.gcn", verbose=False, on_unused_parameters="ignore")
        A, B, C, D = mod.linearize_model(verbose=False, steady_state_kwargs={"verbose": False, "progressbar": False})

        T, R, result, _log_norm = solve_policy_function_with_cycle_reduction(
            A, B, C, D, max_iter=100, tol=1e-8, verbose=False
        )

        assert T is None
        assert R is None
        assert result == "Iteration on all matrices failed to converge"

    def test_diverging_iteration_reports_failure_without_arithmetic_warnings(self):
        rng = np.random.default_rng(0)
        n = 6
        A0 = rng.normal(size=(n, n)) * 50
        A1 = np.eye(n) * 1e-3
        A2 = rng.normal(size=(n, n)) * 50

        X, res, result, _log_norm = cycle_reduction_numpy(A0, A1, A2, max_iter=200, tol=1e-8)

        assert X is None
        assert res is None
        assert result == "Iteration on all matrices failed to converge"


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
            pt.tensor(name=name, shape=x.shape) for name, x in zip("ABCD", [A, B, C, D], strict=True)
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

        T_gs, R_gs, _ = gensys_pt(A_pt, B_pt, C_pt, D_pt, tol=1e-8)
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


class TestNumbaBackend:
    @pytest.mark.parametrize(
        ("op", "dtype"),
        [
            (gensys_pt, "float64"),
            (gensys_pt, "float32"),
            (cycle_reduction_pt, "float64"),
            (cycle_reduction_pt, "float32"),
            (scan_cycle_reduction, "float64"),
            (scan_cycle_reduction, "float32"),
        ],
        ids=[
            "gensys-f64",
            "gensys-f32",
            "cycle_reduction-f64",
            "cycle_reduction-f32",
            "scan_cycle_reduction-f64",
            "scan_cycle_reduction-f32",
        ],
    )
    @pytest.mark.parametrize("static_shape", [True, False], ids=["static_shape", "unknown_shape"])
    def test_matches_python_backend(self, op, dtype, static_shape):
        """The njit kernels agree with the Python Ops, whatever the input dtype and whether the shape is static."""
        mod = load_and_cache_model("one_block_1_ss.gcn")
        A, B, C, D = [
            np.ascontiguousarray(x, dtype="float64")
            for x in mod.linearize_model(verbose=False, steady_state_kwargs={"verbose": False, "progressbar": False})
        ]

        inputs = [
            pt.tensor(name=name, shape=x.shape if static_shape else (None, None), dtype=dtype)
            for name, x in zip("ABCD", [A, B, C, D], strict=True)
        ]
        T, R, *_ = op(*inputs)

        f_py = pytensor.function(inputs, [T, R], mode="FAST_RUN")
        f_numba = pytensor.function(inputs, [T, R], mode="NUMBA")

        values = [x.astype(dtype) for x in [A, B, C, D]]
        T_py, R_py = f_py(*values)
        T_numba, R_numba = f_numba(*values)

        tol = 1e-4 if dtype == "float32" else 1e-8
        assert_allclose(T_numba, T_py, atol=tol, rtol=tol)
        assert_allclose(R_numba, R_py, atol=tol, rtol=tol)
        assert_allclose(A + B @ T_py + C @ T_py @ T_py, 0.0, atol=tol)


class TestCheckBKCondition:
    @staticmethod
    def _system(gcn_file):
        if gcn_file == "pert_fails.gcn":
            mod = model_from_gcn(f"tests/_resources/test_gcns/{gcn_file}", verbose=False, on_unused_parameters="ignore")
        else:
            mod = load_and_cache_model(gcn_file)
        return mod.linearize_model(verbose=False, steady_state_kwargs={"verbose": False, "progressbar": False})

    @pytest.mark.parametrize(
        ("gcn_file", "satisfied"), [("one_block_1_ss.gcn", True), ("pert_fails.gcn", False)], ids=["pass", "fail"]
    )
    def test_return_value_selects_output(self, gcn_file, satisfied):
        A, B, C, D = self._system(gcn_file)

        assert check_bk_condition(A, B, C, D, return_value="bool", verbose=False) is satisfied
        assert check_bk_condition(A, B, C, D, return_value=None, verbose=False) is None

    def test_on_failure_raise_names_the_failure(self):
        A, B, C, D = self._system("pert_fails.gcn")
        with pytest.raises(ValueError, match=r"NOT satisfied\. No unique solution"):
            check_bk_condition(A, B, C, D, on_failure="raise", return_value=None, verbose=False)

    def test_on_failure_raise_is_silent_when_satisfied(self):
        A, B, C, D = self._system("one_block_1_ss.gcn")
        assert check_bk_condition(A, B, C, D, on_failure="raise", return_value="bool", verbose=False)

    @pytest.mark.parametrize(
        ("gcn_file", "satisfied"), [("one_block_1_ss.gcn", True), ("pert_fails.gcn", False)], ids=["pass", "fail"]
    )
    def test_symbolic_check_agrees_with_numpy(self, gcn_file, satisfied):
        A, B, C, D = self._system(gcn_file)
        lead_var_idx = np.flatnonzero(np.abs(C).sum(axis=0) > 1e-8)

        A_pt, B_pt, C_pt, D_pt = (
            pt.tensor(name=name, shape=x.shape) for name, x in zip("ABCD", [A, B, C, D], strict=True)
        )
        outputs = check_bk_condition_pt(A_pt, B_pt, C_pt, D_pt, lead_var_idx)
        bk_satisfied, n_forward, n_unstable = pytensor.function([A_pt, B_pt, C_pt], outputs)(A, B, C)

        assert bk_satisfied == satisfied
        assert n_forward == len(lead_var_idx)
        assert (n_forward == n_unstable) == satisfied


class TestBKConditionGradients:
    @pytest.mark.include_nk
    def test_bk_potential_hessp_builds(self):
        """Detaching the BK eigenvalues must keep the step-function potential twice-differentiable."""
        mod = load_and_cache_model("full_nk.gcn")
        A, B, C, D = [
            np.ascontiguousarray(x, dtype="float64")
            for x in mod.linearize_model(
                verbose=False,
                steady_state_kwargs={"verbose": False, "progressbar": False},
            )
        ]
        lead_var_idx = np.flatnonzero(np.abs(C).sum(axis=0) > 1e-8)

        A_pt, B_pt, C_pt, D_pt = (
            pt.tensor(name=name, shape=x.shape) for name, x in zip("ABCD", [A, B, C, D], strict=True)
        )

        bk_satisfied, _, _ = check_bk_condition_pt(A_pt, B_pt, C_pt, D_pt, lead_var_idx)
        potential = pt.switch(pt.eq(bk_satisfied, 0.0), -np.inf, 0.0)

        # A differentiable term keeps the inputs connected to the cost outside the disconnected BK path, as the real
        # likelihood does.
        cost = potential + (A_pt**2).sum() + (B_pt**2).sum() + (C_pt**2).sum()

        g = pt.grad(cost, [A_pt, B_pt, C_pt])
        ps = [pt.tensor(name=f"p_{n}", shape=x.shape) for n, x in zip("ABC", [A, B, C], strict=True)]
        g_dot_p = sum((gi * pi).sum() for gi, pi in zip(g, ps, strict=True))
        hp = pt.grad(g_dot_p, [A_pt, B_pt, C_pt])

        f = pytensor.function([A_pt, B_pt, C_pt, D_pt, *ps], hp, on_unused_input="ignore")
        hp_vals = f(A, B, C, D, np.ones_like(A), np.ones_like(B), np.ones_like(C))

        # The BK potential is a step function, so the only contribution to the Hessian comes from the quadratic term:
        # 2 * p for each connected input.
        for hp_val, p_val in zip(hp_vals, [np.ones_like(A), np.ones_like(B), np.ones_like(C)], strict=True):
            assert_allclose(hp_val, 2.0 * p_val, atol=1e-8)
