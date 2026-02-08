import re
import unittest

from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import (
    ControlVariableNotFoundException,
    DynamicCalibratingEquationException,
    MultipleObjectiveFunctionsException,
    OptimizationProblemNotDefinedException,
)
from gEconpy.model.block import Block
from gEconpy.parser import constants
from gEconpy.parser.loader import load_gcn_file, load_gcn_string
from gEconpy.parser.preprocessor import preprocess
from gEconpy.parser.transform.to_block import ast_block_to_block
from gEconpy.utilities import set_equality_equals_zero, unpack_keys_and_values

ROOT = Path(__file__).parent.parent.absolute()


def get_block_from_string(gcn_string: str, block_name: str = "HOUSEHOLD") -> Block:
    """Parse a GCN string and return the specified block (already solved)."""
    result = load_gcn_string(gcn_string)
    return result["block_dict"][block_name]


def get_unsolved_block_from_string(gcn_string: str, block_name: str = "HOUSEHOLD") -> Block:
    """Parse a GCN string, build the block, but don't solve optimization."""
    result = preprocess(gcn_string, validate=True)
    for ast_block in result.ast.blocks:
        if ast_block.name == block_name:
            return ast_block_to_block(ast_block, result.assumptions)
    raise KeyError(f"Block {block_name} not found")


@pytest.fixture
def rng():
    return np.random.default_rng()


class IncompleteBlockDefinitionTests(unittest.TestCase):
    def test_raises_if_controls_missing(self):
        test_file = """
            block HOUSEHOLD
            {
                objective
                {
                    U[] = u[] + beta * E[][U[1]];
                };
            };
            """

        with self.assertRaises(OptimizationProblemNotDefinedException):
            get_block_from_string(test_file)

    def test_raises_if_objective_missing(self):
        test_file = """
            block HOUSEHOLD
            {
                controls
                {
                    K[], I[], C[], L[];
                };
            };
            """

        with self.assertRaises(OptimizationProblemNotDefinedException):
            get_block_from_string(test_file)

    def test_raises_if_multiple_objective(self):
        test_file = """
            block HOUSEHOLD
            {
                objective
                {
                    U[] = u[] + beta * E[][U[1]];
                    C[] = a[] + b[];
                };
                controls
                {
                    K[], I[], C[], L[];
                };
            };
            """

        with self.assertRaises(MultipleObjectiveFunctionsException):
            get_block_from_string(test_file)

    def test_raises_if_controls_not_found(self):
        test_file = """
            block HOUSEHOLD
            {
                objective
                {
                    U[] = u[] + beta * E[][U[1]];
                };
                controls
                {
                    Z[];
                };
            };
            """

        with self.assertRaises(ControlVariableNotFoundException):
            get_block_from_string(test_file)

    def test_block_parser_handles_empty_block(self):
        test_file = """
            block HOUSEHOLD
            {
                definitions
                {

                };
                identities
                {
                    Y[] = C[] + I[];
                };
            };
            """
        block = get_block_from_string(test_file)
        # Empty definitions should be None or an empty dict
        self.assertTrue(block.definitions is None or len(block.definitions) == 0)

    def test_non_ss_var_in_calibration_raises(self):
        test_file = """
            block HOUSEHOLD
            {
                calibration
                {
                    Y[ss] / K[] = 0.33 -> alpha;
                };
            };
            """

        self.assertRaises(DynamicCalibratingEquationException, get_block_from_string, test_file)

    def test_function_of_variables_in_calibration_raises(self):
        test_file = """
            block HOUSEHOLD
            {
                calibration
                {
                    beta = 0.99;
                    alpha = beta * Y[];
                };
            };
            """

        self.assertRaises(ValueError, get_block_from_string, test_file)

    def test_lagrange_multiplier_in_objective(self):
        test_file = """
            block HOUSEHOLD
            {
                definitions
                {
                    u[] = log(C[]);
                };

                objective
                {
                    U[] = u[] + beta * E[][U[1]] : lambda[];
                };

                controls
                {
                    C[], K[];
                };

                constraints
                {
                    Y[] = K[-1] ^ alpha;
                    K[] = (1 - delta) * K[-1];
                    C[] = r[] * K[-1];
                };

                calibration
                {
                    alpha = 0.33;
                    delta = 0.035;
                    beta = 0.99;
                };
            };
            """

        block = get_unsolved_block_from_string(test_file)

        with self.assertRaises(NotImplementedError):
            block.solve_optimization()


def test_invalid_decorator_raises():
    test_file = """
        block HOUSEHOLD
        {
            objective
            {
                @exclude
                U[] = u[] + beta * E[][U[1]] : lambda[];
            };

            controls
            {
                u[];
            };
        };
        """

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Equation Eq(U_t, beta*U_t+1 + u_t) in objective block of HOUSEHOLD has an invalid decorator: exclude."
        ),
    ):
        get_block_from_string(test_file)


@pytest.fixture
def block():
    result = load_gcn_file(ROOT / "_resources" / "test_gcns" / "one_block_2.gcn")
    return result["block_dict"]["HOUSEHOLD"]


class TestBlockCases:
    def test_string_repr(self, block):
        assert (
            str(block) == f"{block.name} Block of {block.n_equations} equations, initialized: "
            f"{block.initialized}, "
            f"solved: {block.system_equations is not None}"
        )

    def test_html_repr(self, block):
        html_string = block.__html_repr__()
        assert "Block: HOUSEHOLD" in html_string
        assert "<summary>Definitions</summary>" in html_string
        assert "<summary>Identities</summary>" in html_string
        assert "<summary>Objective</summary>" in html_string
        assert "<summary>Controls</summary>" in html_string
        assert "<summary>Calibration</summary>" in html_string
        assert "class='block-info'" in html_string

    def test_attributes_present(self, block):
        for component in constants.BLOCK_COMPONENTS:
            assert getattr(block, component.lower()) is not None

    def test_eq_number(self, block):
        assert block.n_equations == 14

    def test_variable_list_parsing(self, block):
        for variable in block.controls:
            assert isinstance(variable, TimeAwareSymbol)
        assert len(block.controls) == 5

        for variable in block.shocks:
            assert isinstance(variable, TimeAwareSymbol)
        assert len(block.shocks) == 1

    def test_lagrange_parsing(self, block):
        n_nones = [0 if x is None else 1 for x in list(block.multipliers.values())]
        assert sum(n_nones) == 2
        assert block.multipliers[3] == TimeAwareSymbol("lambda", 0)
        assert block.multipliers[4] == TimeAwareSymbol("q", 0)

    def test_extract_discount_factor_on_Bellman_eq(self, block):
        df = block._get_discount_factor()
        assert df.name == "beta"

    def test_extract_discount_factor_on_static_eq(self, block):
        PI = TimeAwareSymbol("Pi", 0)
        P = TimeAwareSymbol("P", 0)
        Y = TimeAwareSymbol("Y", 0)
        r = TimeAwareSymbol("r", 0)
        w = TimeAwareSymbol("w", 0)
        L = TimeAwareSymbol("L", 0)
        K = TimeAwareSymbol("K", 0)

        block.objective = {0: sp.Eq(PI, P * Y - r * K - w * L)}
        df = block._get_discount_factor()
        assert np.allclose(float(df), 1.0)

    def test_extract_discount_factor_on_lagged_eq(self, block):
        PI = TimeAwareSymbol("Pi", 0)
        P = TimeAwareSymbol("P", 0)
        Y = TimeAwareSymbol("Y", 0)
        r = TimeAwareSymbol("r", 0)
        w = TimeAwareSymbol("w", 0)
        L = TimeAwareSymbol("L", 0)
        K = TimeAwareSymbol("K", -1)

        block.objective = {0: sp.Eq(PI, P * Y - r * K - w * L)}
        df = block._get_discount_factor()
        assert np.allclose(float(df), 1)

    def test_household_lagrangian_function(self, block):
        U = TimeAwareSymbol("U", 1)
        Y = TimeAwareSymbol("Y", 0, positive=True)
        C = TimeAwareSymbol("C", 0, positive=True)
        I = TimeAwareSymbol("I", 0, positive=True)
        K = TimeAwareSymbol("K", 0, positive=True)
        L = TimeAwareSymbol("L", 0, positive=True)
        A = TimeAwareSymbol("A", 0, positive=True)
        lamb = TimeAwareSymbol("lambda", 0)
        lamb_H_1 = TimeAwareSymbol("lambda__H_1", 0)
        q = TimeAwareSymbol("q", 0)

        alpha, beta, delta, theta, tau = sp.symbols(["alpha", "beta", "delta", "theta", "tau"], positive=True)
        Theta, zeta = sp.symbols(["Theta", "zeta"])

        utility = (C**theta * (1 - L) ** (1 - theta)) ** (1 - tau) / (1 - tau)
        mkt_clearing = C + I - Y
        production = Y - A * K**alpha * L ** (1 - alpha) - (Theta + zeta)
        law_motion_K = K - (1 - delta) * K.step_backward() - I

        answer = beta * U + utility - lamb * mkt_clearing - q * law_motion_K - lamb_H_1 * production

        lagrangian = block._build_lagrangian()
        assert (lagrangian - answer).simplify().evalf() == 0

    def test_Household_FOC(self, block, rng):
        block.solve_optimization(try_simplify=False)
        _, identities = unpack_keys_and_values(block.identities)
        _, objective = unpack_keys_and_values(block.objective)
        _, definitions = unpack_keys_and_values(block.definitions)
        sub_dict = {eq.lhs: eq.rhs for eq in definitions}
        objective = set_equality_equals_zero(objective[0].subs(sub_dict))

        assert all(set_equality_equals_zero(eq) in block.system_equations for eq in identities)
        assert objective in block.system_equations

        U = TimeAwareSymbol("U", 1)
        Y = TimeAwareSymbol("Y", 0, positive=True)
        C = TimeAwareSymbol("C", 0, positive=True)
        I = TimeAwareSymbol("I", 0, positive=True)
        K = TimeAwareSymbol("K", 0, positive=True)
        L = TimeAwareSymbol("L", 0, positive=True)
        A = TimeAwareSymbol("A", 0, positive=True)
        lamb = TimeAwareSymbol("lambda", 0)
        lamb_H_1 = TimeAwareSymbol("lambda__H_1", 0)
        q = TimeAwareSymbol("q", 0)
        eps = TimeAwareSymbol("epsilon", 0)

        alpha, beta, delta, theta, tau, rho = sp.symbols(
            ["alpha", "beta", "delta", "theta", "tau", "rho"], positive=True
        )
        Theta, zeta = sp.symbols("Theta, zeta")

        all_variables = [
            U,
            U.step_backward(),
            Y,
            C,
            I,
            K,
            K.step_backward(),
            L,
            A,
            A.step_backward(),
            lamb,
            lamb_H_1,
            q,
            q.step_forward(),
            alpha,
            beta,
            delta,
            theta,
            tau,
            rho,
            eps,
            L.to_ss(),
            K.to_ss(),
        ]

        sub_dict = dict(zip(all_variables, rng.uniform(0, 1, size=len(all_variables)), strict=False))
        sub_dict[Theta] = 0
        sub_dict[zeta] = 0

        dL_dC = (C**theta * (1 - L) ** (1 - theta)) ** (-tau) * C ** (theta - 1) * (1 - L) ** (1 - theta) * theta - lamb

        dL_dL = (C**theta * (1 - L) ** (1 - theta)) ** (-tau) * C**theta * (1 - L) ** (-theta) * (
            1 - theta
        ) * -1 + lamb_H_1 * (1 - alpha) * A * K**alpha * L ** (-alpha)
        dL_dK = lamb_H_1 * A * alpha * K ** (alpha - 1) * L ** (1 - alpha) - q + beta * (1 - delta) * q.step_forward()
        dL_dI = -lamb + q

        subbed_system = [np.float32(eq.subs(sub_dict)) for eq in block.system_equations]

        for solution in [dL_dC, dL_dL, dL_dK, dL_dI]:
            assert np.float32(solution.subs(sub_dict)) in subbed_system

    def test_firm_block_lagrange_parsing(self):
        result = load_gcn_file(ROOT / "_resources" / "test_gcns" / "rbc_2_block.gcn")
        block = result["block_dict"]["FIRM"]

        Y = TimeAwareSymbol("Y", 0)
        K = TimeAwareSymbol("K", -1)
        L = TimeAwareSymbol("L", 0)
        A = TimeAwareSymbol("A", 0)
        r = TimeAwareSymbol("r", 0)
        w = TimeAwareSymbol("w", 0)
        P = TimeAwareSymbol("P", 0)
        alpha, _rho = sp.symbols(["alpha", "rho"])

        tc = -(r * K + w * L)
        prod = Y - A * K**alpha * L ** (1 - alpha)
        L = tc - P * prod

        assert (block._build_lagrangian() - L).simplify() == 0

    def test_firm_FOC(self, rng):
        result = load_gcn_file(ROOT / "_resources" / "test_gcns" / "rbc_2_block.gcn")
        firm_block = result["block_dict"]["FIRM"]
        firm_block.solve_optimization()

        Y = TimeAwareSymbol("Y", 0)
        TC = TimeAwareSymbol("TC", 0)
        K = TimeAwareSymbol("K", -1)
        L = TimeAwareSymbol("L", 0)
        A = TimeAwareSymbol("A", 0)
        r = TimeAwareSymbol("r", 0)
        w = TimeAwareSymbol("w", 0)
        P = TimeAwareSymbol("P", 0)
        epsilon = TimeAwareSymbol("epsilon_A", 0)
        alpha, rho = sp.symbols(["alpha", "rho_A"])

        all_variables = [
            Y,
            TC,
            K,
            L,
            A,
            A.step_backward(),
            P,
            r,
            w,
            alpha,
            rho,
            epsilon,
        ]

        sub_dict = dict(zip(all_variables, rng.uniform(0, 1, size=len(all_variables)), strict=False))

        dL_dK = -r + P * A * alpha * K ** (alpha - 1) * L ** (1 - alpha)
        dL_dL = -w + P * A * (1 - alpha) * K**alpha * L ** (-alpha)

        subbed_system = [eq.subs(sub_dict) for eq in firm_block.system_equations]

        for solution in [dL_dK, dL_dL]:
            assert solution.subs(sub_dict) in subbed_system

    def test_get_param_dict_and_calibrating_equations(self, block):
        block.solve_optimization(try_simplify=False)

        _alpha, theta, beta, delta, tau, rho = sp.symbols(
            ["alpha", "theta", "beta", "delta", "tau", "rho"], positive=True
        )
        K = TimeAwareSymbol("K", 0, positive=True).to_ss()
        L = TimeAwareSymbol("L", 0, positive=True).to_ss()

        answer = {theta: 0.357, beta: 1 / 1.01, delta: 0.02, tau: 2, rho: 0.95}
        assert all(key in block.param_dict for key in answer)

        for key in block.param_dict:
            np.testing.assert_allclose(answer[key], block.param_dict.values_to_float()[key])

        # Compare by name since params_to_calibrate may have different symbol assumptions
        assert [str(p) for p in block.params_to_calibrate] == ["alpha"]

        # Get the actual symbols from the block for equation comparison
        actual_alpha = block.params_to_calibrate[0]
        actual_L_ss = next(s for s in block.calibrating_equations[0].free_symbols if str(s) == "L_ss")
        actual_K_ss = next(s for s in block.calibrating_equations[0].free_symbols if str(s) == "K_ss")

        # GCN: L[ss] / K[ss] = 0.36 -> alpha
        # Convention: alpha = RHS - LHS = 0.36 - L_ss/K_ss
        # So: param - calib_eq = alpha - (0.36 - L_ss/K_ss) = alpha - 0.36 + L_ss/K_ss
        calibrating_eqs = [actual_alpha - 0.36 + actual_L_ss / actual_K_ss]

        for i, eq in enumerate(calibrating_eqs):
            assert eq.simplify() == (block.params_to_calibrate[i] - block.calibrating_equations[i]).simplify()

    def test_deterministic_relationships(self, block):
        assert len(block.deterministic_relationships) == 2
        assert len(block.deterministic_params) == 2

        assert [x.name for x in block.deterministic_params] == ["Theta", "zeta"]
        answers = [3 + 1 / 1.01 * 0.95, -np.log(0.357)]
        for eq, answer in zip(block.deterministic_relationships, answers, strict=False):
            np.testing.assert_allclose(float(eq.subs(block.param_dict).evalf()), answer)

    def test_variable_list(self, block):
        block.solve_optimization(try_simplify=False)
        assert {x.base_name for x in block.variables} == {
            "A",
            "C",
            "I",
            "K",
            "L",
            "U",
            "Y",
            "lambda",
            "q",
            "lambda__H_1",
        }
        assert {x.base_name for x in block.shocks} == {"epsilon"}


def test_block_with_exlcuded_equation():
    result = load_gcn_file(ROOT / "_resources" / "test_gcns" / "rbc_with_excluded.gcn")
    block = result["block_dict"]["HOUSEHOLD"]
    block.solve_optimization()

    # 6 equations are 4 controls, 1 objective, 1 constraint (excluding the excluded equation)
    assert len(block.system_equations) == 6


class TestBlockFromSympy:
    def test_from_sympy_creates_valid_block(self):
        C = TimeAwareSymbol("C", 0)
        Y = TimeAwareSymbol("Y", 0)

        identities = {0: sp.Eq(Y, C)}
        equation_flags = {0: {}}

        block = Block.from_sympy(
            name="TEST",
            identities=identities,
            equation_flags=equation_flags,
        )

        assert block.name == "TEST"
        assert block.initialized is True
        assert block.identities == identities

    def test_from_sympy_matches_dict_constructor(self):
        test_file = """
            block HOUSEHOLD
            {
                identities
                {
                    Y[] = C[] + I[];
                    K[] = I[] + (1 - delta) * K[-1];
                };

                calibration
                {
                    delta = 0.02;
                };
            };
            """

        loaded_block = get_block_from_string(test_file)
        loaded_block.solve_optimization()

        # Create sympy objects for from_sympy constructor
        Y = TimeAwareSymbol("Y", 0)
        C = TimeAwareSymbol("C", 0)
        I = TimeAwareSymbol("I", 0)
        K = TimeAwareSymbol("K", 0)
        K_lag = TimeAwareSymbol("K", -1)
        delta = sp.Symbol("delta")

        identities = {
            0: sp.Eq(Y, C + I),
            1: sp.Eq(K, I + (1 - delta) * K_lag),
        }
        calibration = {
            2: sp.Eq(delta, sp.Float(0.02)),
        }
        equation_flags = {0: {}, 1: {}, 2: {"is_calibrating": False}}

        new_block = Block.from_sympy(
            name="HOUSEHOLD",
            identities=identities,
            calibration=calibration,
            equation_flags=equation_flags,
        )
        new_block.solve_optimization()

        # Compare key attributes
        assert loaded_block.name == new_block.name
        assert len(loaded_block.system_equations) == len(new_block.system_equations)
        assert {v.base_name for v in loaded_block.variables} == {v.base_name for v in new_block.variables}

        # Compare parameters
        assert set(loaded_block.param_dict.keys()) == set(new_block.param_dict.keys())
        for key in loaded_block.param_dict:
            assert float(loaded_block.param_dict[key]) == float(new_block.param_dict[key])

    def test_from_sympy_with_optimization_problem(self):
        U = TimeAwareSymbol("U", 0)
        U_next = TimeAwareSymbol("U", 1)
        C = TimeAwareSymbol("C", 0)
        L = TimeAwareSymbol("L", 0)
        w = TimeAwareSymbol("w", 0)
        lambda_ = TimeAwareSymbol("lambda", 0)
        beta = sp.Symbol("beta")

        objective = {0: sp.Eq(U, sp.log(C) - L + beta * U_next)}
        constraints = {1: sp.Eq(C, w * L)}
        controls = [C, L]
        # Multipliers should map constraint indices to multiplier symbols
        # Objective index (0) should map to None
        multipliers = {0: None, 1: lambda_}
        equation_flags = {0: {}, 1: {}}

        block = Block.from_sympy(
            name="HOUSEHOLD",
            objective=objective,
            constraints=constraints,
            controls=controls,
            multipliers=multipliers,
            equation_flags=equation_flags,
        )

        assert block.initialized is True
        assert block.controls == controls
        assert block.objective == objective
        assert block.constraints == constraints

        # Solve should work
        block.solve_optimization()
        assert len(block.system_equations) > 0
