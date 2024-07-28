import os
import unittest

from pathlib import Path

import numpy as np
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions.exceptions import (
    ControlVariableNotFoundException,
    DynamicCalibratingEquationException,
    MultipleObjectiveFunctionsException,
    OptimizationProblemNotDefinedException,
)
from gEconpy.model.block import Block
from gEconpy.parser import constants, file_loaders, gEcon_parser
from gEconpy.shared.utilities import set_equality_equals_zero, unpack_keys_and_values

ROOT = Path(__file__).parent.absolute()


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

        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict, options, tryreduce, assumptions = (
            gEcon_parser.split_gcn_into_dictionaries(parser_output)
        )
        block_dict = gEcon_parser.parsed_block_to_dict(block_dict["HOUSEHOLD"])

        self.assertRaises(
            OptimizationProblemNotDefinedException, Block, "HOUSEHOLD", block_dict
        )

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

        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict, options, tryreduce, assumptions = (
            gEcon_parser.split_gcn_into_dictionaries(parser_output)
        )
        block_dict = gEcon_parser.parsed_block_to_dict(block_dict["HOUSEHOLD"])

        self.assertRaises(
            OptimizationProblemNotDefinedException, Block, "HOUSEHOLD", block_dict
        )

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

        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict, options, tryreduce, assumptions = (
            gEcon_parser.split_gcn_into_dictionaries(parser_output)
        )
        block_dict = gEcon_parser.parsed_block_to_dict(block_dict["HOUSEHOLD"])

        self.assertRaises(
            MultipleObjectiveFunctionsException, Block, "HOUSEHOLD", block_dict
        )

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

        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict, options, tryreduce, assumptions = (
            gEcon_parser.split_gcn_into_dictionaries(parser_output)
        )
        block_dict = gEcon_parser.parsed_block_to_dict(block_dict["HOUSEHOLD"])

        self.assertRaises(
            ControlVariableNotFoundException, Block, "HOUSEHOLD", block_dict
        )

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
        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict, options, tryreduce, assumptions = (
            gEcon_parser.split_gcn_into_dictionaries(parser_output)
        )
        block_dict = gEcon_parser.parsed_block_to_dict(block_dict["HOUSEHOLD"])

        block = Block("HOUSEHOLD", block_dict)
        self.assertTrue(len(block.definitions) == 0)

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

        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict, options, tryreduce, assumptions = (
            gEcon_parser.split_gcn_into_dictionaries(parser_output)
        )
        block_dict = gEcon_parser.parsed_block_to_dict(block_dict["HOUSEHOLD"])

        self.assertRaises(
            DynamicCalibratingEquationException, Block, "HOUSEHOLD", block_dict
        )

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

        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict, options, tryreduce, assumptions = (
            gEcon_parser.split_gcn_into_dictionaries(parser_output)
        )
        block_dict = gEcon_parser.parsed_block_to_dict(block_dict["HOUSEHOLD"])

        self.assertRaises(ValueError, Block, "HOUSEHOLD", block_dict)

    def test_multiple_leads_in_objective_raises(self):
        test_file = """
            block HOUSEHOLD
            {
                objective
                {
                    U[] = u[] + X[] + beta * E[][U[1] + X[1]];
                };

                controls
                {
                    X[];
                };
            };
            """

        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict, options, tryreduce, assumptions = (
            gEcon_parser.split_gcn_into_dictionaries(parser_output)
        )
        block_dict = gEcon_parser.parsed_block_to_dict(block_dict["HOUSEHOLD"])
        block = Block("HOUSEHOLD", block_dict)

        with self.assertRaises(ValueError) as error:
            block.solve_optimization()
        error_msg = error.exception
        self.assertEqual(
            str(error_msg),
            "Block HOUSEHOLD has multiple t+1 variables in the Bellman equation, this is not "
            "currently supported. Rewrite the equation in the form X[] = a[] + b * E[][X[1]], "
            "where a[] is the instantaneous value function at time t, defined in the "
            '"definitions" component of the block.',
        )

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

        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict, options, tryreduce, assumptions = (
            gEcon_parser.split_gcn_into_dictionaries(parser_output)
        )
        block_dict = gEcon_parser.parsed_block_to_dict(block_dict["HOUSEHOLD"])
        block = Block("HOUSEHOLD", block_dict)

        with self.assertRaises(NotImplementedError):
            block.solve_optimization()


class BlockTestCases(unittest.TestCase):
    def setUp(self):
        test_file = file_loaders.load_gcn(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_2.gcn")
        )
        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict, options, tryreduce, assumptions = (
            gEcon_parser.split_gcn_into_dictionaries(parser_output)
        )
        block_dict = gEcon_parser.parsed_block_to_dict(block_dict["HOUSEHOLD"])

        self.block = Block("HOUSEHOLD", block_dict)

    def test_string_repr(self):
        block_str = str(self.block)

        self.assertEqual(
            block_str,
            f"{self.block.name} Block of {self.block.n_equations} equations, initialized: "
            f"{self.block.initialized}, "
            f"solved: {self.block.system_equations is not None}",
        )

    def test_attributes_present(self):
        for component in constants.BLOCK_COMPONENTS:
            self.assertNotEqual(getattr(self.block, component.lower()), None)

    def test_eq_number(self):
        self.assertEqual(self.block.n_equations, 14)

    def test_variable_list_parsing(self):
        for variable in self.block.controls:
            self.assertIsInstance(variable, TimeAwareSymbol)
        self.assertEqual(len(self.block.controls), 5)

        for variable in self.block.shocks:
            self.assertIsInstance(variable, TimeAwareSymbol)
        self.assertEqual(len(self.block.shocks), 1)

    def test_lagrange_parsing(self):
        n_nones = [0 if x is None else 1 for x in list(self.block.multipliers.values())]
        self.assertEqual(sum(n_nones), 2)
        self.assertEqual(self.block.multipliers[3], TimeAwareSymbol("lambda", 0))
        self.assertEqual(self.block.multipliers[4], TimeAwareSymbol("q", 0))

    def test_extract_discount_factor_on_Bellman_eq(self):
        df = self.block._get_discount_factor()
        self.assertEqual(df.name, "beta")

    def test_extract_discount_factor_on_static_eq(self):
        PI = TimeAwareSymbol("Pi", 0)
        P = TimeAwareSymbol("P", 0)
        Y = TimeAwareSymbol("Y", 0)
        r = TimeAwareSymbol("r", 0)
        w = TimeAwareSymbol("w", 0)
        L = TimeAwareSymbol("L", 0)
        K = TimeAwareSymbol("K", 0)

        self.block.objective = {0: sp.Eq(PI, P * Y - r * K - w * L)}
        df = self.block._get_discount_factor()
        self.assertEqual(df, 1)

    def test_extract_discount_factor_on_lagged_eq(self):
        PI = TimeAwareSymbol("Pi", 0)
        P = TimeAwareSymbol("P", 0)
        Y = TimeAwareSymbol("Y", 0)
        r = TimeAwareSymbol("r", 0)
        w = TimeAwareSymbol("w", 0)
        L = TimeAwareSymbol("L", 0)
        K = TimeAwareSymbol("K", -1)

        self.block.objective = {0: sp.Eq(PI, P * Y - r * K - w * L)}
        df = self.block._get_discount_factor()
        self.assertEqual(df, 1)

    def test_household_lagrangian_function(self):
        U = TimeAwareSymbol("U", 1)
        Y = TimeAwareSymbol("Y", 0)
        C = TimeAwareSymbol("C", 0)
        I = TimeAwareSymbol("I", 0)
        K = TimeAwareSymbol("K", 0)
        L = TimeAwareSymbol("L", 0)
        A = TimeAwareSymbol("A", 0)
        lamb = TimeAwareSymbol("lambda", 0)
        lamb_H_1 = TimeAwareSymbol("lambda__H_1", 0)
        q = TimeAwareSymbol("q", 0)

        alpha, beta, delta, theta, tau, Theta, zeta = sp.symbols(
            ["alpha", "beta", "delta", "theta", "tau", "Theta", "zeta"]
        )

        utility = (C**theta * (1 - L) ** (1 - theta)) ** (1 - tau) / (1 - tau)
        mkt_clearing = C + I - Y
        production = Y - A * K**alpha * L ** (1 - alpha) - (Theta + zeta)
        law_motion_K = K - (1 - delta) * K.step_backward() - I

        answer = (
            beta * U
            + utility
            - lamb * mkt_clearing
            - q * law_motion_K
            - lamb_H_1 * production
        )

        L = self.block._build_lagrangian()
        self.assertEqual((L - answer).simplify(), 0)

    def test_Household_FOC(self):
        self.block.solve_optimization(try_simplify=False)
        _, identities = unpack_keys_and_values(self.block.identities)
        _, objective = unpack_keys_and_values(self.block.objective)
        _, definitions = unpack_keys_and_values(self.block.definitions)
        sub_dict = {eq.lhs: eq.rhs for eq in definitions}
        objective = set_equality_equals_zero(objective[0].subs(sub_dict))

        self.assertEqual(
            all(
                [
                    set_equality_equals_zero(eq) in self.block.system_equations
                    for eq in identities
                ]
            ),
            True,
        )
        self.assertIn(objective, self.block.system_equations)

        U = TimeAwareSymbol("U", 1)
        Y = TimeAwareSymbol("Y", 0)
        C = TimeAwareSymbol("C", 0)
        I = TimeAwareSymbol("I", 0)
        K = TimeAwareSymbol("K", 0)
        L = TimeAwareSymbol("L", 0)
        A = TimeAwareSymbol("A", 0)
        lamb = TimeAwareSymbol("lambda", 0)
        lamb_H_1 = TimeAwareSymbol("lambda__H_1", 0)
        q = TimeAwareSymbol("q", 0)
        eps = TimeAwareSymbol("epsilon", 0)

        alpha, beta, delta, theta, tau, rho = sp.symbols(
            ["alpha", "beta", "delta", "theta", "tau", "rho"]
        )
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

        sub_dict = dict(
            zip(all_variables, np.random.uniform(0, 1, size=len(all_variables)))
        )

        # These are extraneous parameters used to test deterministic relationships. We can ignore them for the
        # purpose of this test.
        Theta, zeta = sp.symbols("Theta, zeta")
        sub_dict[Theta] = 0
        sub_dict[zeta] = 0

        dL_dC = (C**theta * (1 - L) ** (1 - theta)) ** (-tau) * C ** (theta - 1) * (
            1 - L
        ) ** (1 - theta) * theta - lamb

        dL_dL = (C**theta * (1 - L) ** (1 - theta)) ** (-tau) * C**theta * (1 - L) ** (
            -theta
        ) * (1 - theta) * -1 + lamb_H_1 * (1 - alpha) * A * K**alpha * L ** (-alpha)
        dL_dK = (
            lamb_H_1 * A * alpha * K ** (alpha - 1) * L ** (1 - alpha)
            - q
            + beta * (1 - delta) * q.step_forward()
        )
        dL_dI = -lamb + q

        subbed_system = [
            np.float32(eq.subs(sub_dict)) for eq in self.block.system_equations
        ]

        for solution in [dL_dC, dL_dL, dL_dK, dL_dI]:
            self.assertIn(np.float32(solution.subs(sub_dict)), subbed_system)

    def test_firm_block_lagrange_parsing(self):
        test_file = file_loaders.load_gcn(
            os.path.join(ROOT, "Test GCNs/Two_Block_RBC_1.gcn")
        )
        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict, options, tryreduce, assumptions = (
            gEcon_parser.split_gcn_into_dictionaries(parser_output)
        )
        block_dict = gEcon_parser.parsed_block_to_dict(block_dict["FIRM"])

        block = Block("FIRM", block_dict)

        Y = TimeAwareSymbol("Y", 0)
        K = TimeAwareSymbol("K", -1)
        L = TimeAwareSymbol("L", 0)
        A = TimeAwareSymbol("A", 0)
        r = TimeAwareSymbol("r", 0)
        w = TimeAwareSymbol("w", 0)
        P = TimeAwareSymbol("P", 0)
        alpha, rho = sp.symbols(["alpha", "rho"])

        tc = -(r * K + w * L)
        prod = Y - A * K**alpha * L ** (1 - alpha)
        L = tc - P * prod

        self.assertEqual((block._build_lagrangian() - L).simplify(), 0)

    def test_firm_FOC(self):
        test_file = file_loaders.load_gcn(
            os.path.join(ROOT, "Test GCNs/Two_Block_RBC_1.gcn")
        )
        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict, options, tryreduce, assumptions = (
            gEcon_parser.split_gcn_into_dictionaries(parser_output)
        )
        firm_dict = gEcon_parser.parsed_block_to_dict(block_dict["FIRM"])

        firm_block = Block("FIRM", firm_dict)
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

        sub_dict = dict(
            zip(all_variables, np.random.uniform(0, 1, size=len(all_variables)))
        )

        dL_dK = -r + P * A * alpha * K ** (alpha - 1) * L ** (1 - alpha)
        dL_dL = -w + P * A * (1 - alpha) * K**alpha * L ** (-alpha)

        subbed_system = [eq.subs(sub_dict) for eq in firm_block.system_equations]

        for solution in [dL_dK, dL_dL]:
            self.assertIn(solution.subs(sub_dict), subbed_system)

    def test_get_param_dict_and_calibrating_equations(self):
        self.block.solve_optimization(try_simplify=False)

        alpha, theta, beta, delta, tau, rho = sp.symbols(
            ["alpha", "theta", "beta", "delta", "tau", "rho"]
        )
        K = TimeAwareSymbol("K", 0).to_ss()
        L = TimeAwareSymbol("L", 0).to_ss()

        answer = {theta: 0.357, beta: 1 / 1.01, delta: 0.02, tau: 2, rho: 0.95}
        self.assertEqual(
            all([key in self.block.param_dict.keys() for key in answer.keys()]), True
        )

        for key in self.block.param_dict:
            np.testing.assert_allclose(
                answer[key], self.block.param_dict.values_to_float()[key]
            )

        assert self.block.params_to_calibrate == [alpha]

        calibrating_eqs = [alpha - L / K + 0.36]

        for i, eq in enumerate(calibrating_eqs):
            self.assertEqual(
                eq.simplify(),
                (
                    self.block.params_to_calibrate[i]
                    - self.block.calibrating_equations[i]
                ).simplify(),
            )

    def test_deterministic_relationships(self):
        self.assertEqual(len(self.block.deterministic_relationships), 2)
        self.assertEqual(len(self.block.deterministic_params), 2)

        self.assertEqual(
            [x.name for x in self.block.deterministic_params], ["Theta", "zeta"]
        )
        answers = [3 + 1 / 1.01 * 0.95, -np.log(0.357)]
        for eq, answer in zip(self.block.deterministic_relationships, answers):
            np.testing.assert_allclose(
                float(eq.subs(self.block.param_dict).evalf()), answer
            )

    def test_variable_list(self):
        self.block.solve_optimization(try_simplify=False)
        self.assertEqual(
            {x.base_name for x in self.block.variables},
            {"A", "C", "I", "K", "L", "U", "Y", "lambda", "q", "lambda__H_1"},
        )
        self.assertEqual({x.base_name for x in self.block.shocks}, {"epsilon"})


if __name__ == "__main__":
    unittest.main()
