import unittest
from gEconpy.classes.block import Block
from gEconpy.shared.utilities import unpack_keys_and_values, set_equality_equals_zero

from gEconpy.parser import file_loaders, gEcon_parser, constants
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol

import sympy as sp
import numpy as np


class BlockTestCases(unittest.TestCase):

    def setUp(self):
        test_file = file_loaders.load_gcn('Test GCNs/One_Block_Simple_2.gcn')
        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict = gEcon_parser.split_gcn_into_block_dictionary(parser_output)
        block_dict = gEcon_parser.parsed_block_to_dict(block_dict['HOUSEHOLD'])

        self.block = Block('HOUSEHOLD', block_dict)

    def test_attributes_present(self):
        for component in constants.BLOCK_COMPONENTS:
            self.assertNotEqual(getattr(self.block, component.lower()), None)

    def test_eq_number(self):
        self.assertEqual(self.block.n_equations, 12)

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
        self.assertEqual(self.block.multipliers[3], TimeAwareSymbol('lambda', 0))
        self.assertEqual(self.block.multipliers[4], TimeAwareSymbol('q', 0))

    def test_extract_discount_factor_on_Bellman_eq(self):
        df = self.block._get_discount_factor()
        self.assertEqual(df.name, 'beta')

    def test_extract_discount_factor_on_static_eq(self):
        PI = TimeAwareSymbol('Pi', 0)
        P = TimeAwareSymbol('P', 0)
        Y = TimeAwareSymbol('Y', 0)
        r = TimeAwareSymbol('r', 0)
        w = TimeAwareSymbol('w', 0)
        L = TimeAwareSymbol('L', 0)
        K = TimeAwareSymbol('K', 0)

        self.block.objective = {0: sp.Eq(PI, P * Y - r * K - w * L)}
        df = self.block._get_discount_factor()
        self.assertEqual(df, 1)

    def test_extract_discount_factor_on_lagged_eq(self):
        PI = TimeAwareSymbol('Pi', 0)
        P = TimeAwareSymbol('P', 0)
        Y = TimeAwareSymbol('Y', 0)
        r = TimeAwareSymbol('r', 0)
        w = TimeAwareSymbol('w', 0)
        L = TimeAwareSymbol('L', 0)
        K = TimeAwareSymbol('K', -1)

        self.block.objective = {0: sp.Eq(PI, P * Y - r * K - w * L)}
        df = self.block._get_discount_factor()
        self.assertEqual(df, 1)

    def test_household_lagrangian_function(self):
        U = TimeAwareSymbol('U', 1)
        Y = TimeAwareSymbol('Y', 0)
        C = TimeAwareSymbol('C', 0)
        I = TimeAwareSymbol('I', 0)
        K = TimeAwareSymbol('K', 0)
        L = TimeAwareSymbol('L', 0)
        A = TimeAwareSymbol('A', 0)
        lamb = TimeAwareSymbol('lambda', 0)
        lamb_H_1 = TimeAwareSymbol('lambda__H_1', 0)
        q = TimeAwareSymbol('q', 0)

        alpha, beta, delta, theta, tau = sp.symbols(['alpha', 'beta', 'delta', 'theta', 'tau'])

        utility = (C ** theta * (1 - L) ** (1 - theta)) ** (1 - tau) / (1 - tau)
        mkt_clearing = C + I - Y
        production = Y - A * K ** alpha * L ** (1 - alpha)
        law_motion_K = K - (1 - delta) * K.step_backward() - I

        answer = beta * U + utility - lamb * mkt_clearing - q * law_motion_K - lamb_H_1 * production

        L = self.block._build_lagrangian()
        self.assertEqual((L - answer).simplify(), 0)

    def test_Household_FOC(self):
        self.block.solve_optimization(try_simplify=False)
        _, identities = unpack_keys_and_values(self.block.identities)
        _, objective = unpack_keys_and_values(self.block.objective)
        _, definitions = unpack_keys_and_values(self.block.definitions)
        sub_dict = {eq.lhs: eq.rhs for eq in definitions}
        objective = set_equality_equals_zero(objective[0].subs(sub_dict))

        self.assertEqual(all([set_equality_equals_zero(eq) in self.block.system_equations for eq in identities]), True)
        self.assertIn(objective, self.block.system_equations)

        U = TimeAwareSymbol('U', 1)
        Y = TimeAwareSymbol('Y', 0)
        C = TimeAwareSymbol('C', 0)
        I = TimeAwareSymbol('I', 0)
        K = TimeAwareSymbol('K', 0)
        L = TimeAwareSymbol('L', 0)
        A = TimeAwareSymbol('A', 0)
        lamb = TimeAwareSymbol('lambda', 0)
        lamb_H_1 = TimeAwareSymbol('lambda__H_1', 0)
        q = TimeAwareSymbol('q', 0)
        eps = TimeAwareSymbol('epsilon', 0)

        alpha, beta, delta, theta, tau, rho = sp.symbols(['alpha', 'beta', 'delta', 'theta', 'tau', 'rho'])
        all_variables = [U, U.step_backward(), Y, C, I, K, K.step_backward(), L, A, A.step_backward(), lamb,
                         lamb_H_1, q, q.step_forward(), alpha, beta, delta, theta, tau, rho, eps, L.to_ss(), K.to_ss()]

        sub_dict = dict(zip(all_variables, np.random.uniform(0, 1, size=len(all_variables))))

        dL_dC = (C ** theta * (1 - L) ** (1 - theta)) ** (-tau) * C ** (theta - 1) * (1 - L) ** (1 - theta) * theta \
            - lamb

        dL_dL = (C ** theta * (1 - L) ** (1 - theta)) ** (-tau) * C ** theta * (1 - L) ** (-theta) * (1 - theta) * -1 \
            + lamb_H_1 * (1 - alpha) * A * K ** alpha * L ** (-alpha)
        dL_dK = lamb_H_1 * A * alpha * K ** (alpha - 1) * L ** (1 - alpha) - q + beta * (1 - delta) * q.step_forward()
        dL_dI = -lamb + q

        subbed_system = [np.float32(eq.subs(sub_dict)) for eq in self.block.system_equations]

        for solution in [dL_dC, dL_dL, dL_dK, dL_dI]:
            self.assertIn(np.float32(solution.subs(sub_dict)), subbed_system)

    def test_firm_block_lagrange_parsing(self):
        test_file = file_loaders.load_gcn('Test GCNs/Two_Block_RBC_1.gcn')
        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict = gEcon_parser.split_gcn_into_block_dictionary(parser_output)
        block_dict = gEcon_parser.parsed_block_to_dict(block_dict['FIRM'])

        block = Block('FIRM', block_dict, prior_dict)

        Y = TimeAwareSymbol('Y', 0)
        K = TimeAwareSymbol('K', -1)
        L = TimeAwareSymbol('L', 0)
        A = TimeAwareSymbol('A', 0)
        r = TimeAwareSymbol('r', 0)
        w = TimeAwareSymbol('w', 0)
        P = TimeAwareSymbol('P', 0)
        alpha, rho = sp.symbols(['alpha', 'rho'])

        tc = -(r * K + w * L)
        prod = Y - A * K ** alpha * L ** (1 - alpha)
        L = tc - P * prod

        self.assertEqual((block._build_lagrangian() - L).simplify(), 0)

    def test_firm_FOC(self):
        test_file = file_loaders.load_gcn('Test GCNs/Two_Block_RBC_1.gcn')
        parser_output, prior_dict = gEcon_parser.preprocess_gcn(test_file)
        block_dict = gEcon_parser.split_gcn_into_block_dictionary(parser_output)
        firm_dict = gEcon_parser.parsed_block_to_dict(block_dict['FIRM'])

        firm_block = Block('FIRM', firm_dict, prior_dict)
        firm_block.solve_optimization()

        Y = TimeAwareSymbol('Y', 0)
        TC = TimeAwareSymbol('TC', 0)
        K = TimeAwareSymbol('K', -1)
        L = TimeAwareSymbol('L', 0)
        A = TimeAwareSymbol('A', 0)
        r = TimeAwareSymbol('r', 0)
        w = TimeAwareSymbol('w', 0)
        P = TimeAwareSymbol('P', 0)
        epsilon = TimeAwareSymbol('epsilon_A', 0)
        alpha, rho = sp.symbols(['alpha', 'rho_A'])

        all_variables = [Y, TC, K, L, A, A.step_backward(), P, r, w,
                         alpha, rho, epsilon]

        sub_dict = dict(zip(all_variables, np.random.uniform(0, 1, size=len(all_variables))))

        dL_dK = -r + P * A * alpha * K ** (alpha - 1) * L ** (1 - alpha)
        dL_dL = -w + P * A * (1 - alpha) * K ** alpha * L ** (-alpha)

        subbed_system = [eq.subs(sub_dict) for eq in firm_block.system_equations]

        for solution in [dL_dK, dL_dL]:
            self.assertIn(solution.subs(sub_dict), subbed_system)

    def test_get_param_dict_and_calibrating_equations(self):

        self.block.solve_optimization(try_simplify=False)

        alpha, theta, beta, delta, tau, rho = sp.symbols(['alpha', 'theta', 'beta', 'delta', 'tau', 'rho'])
        K = TimeAwareSymbol('K', 0).to_ss()
        L = TimeAwareSymbol('L', 0).to_ss()

        answer = {theta: 0.357, beta: 0.99, delta: 0.02, tau: 2, rho: 0.95}
        self.assertEqual(all([key in self.block.param_dict.keys() for key in answer.keys()]), True)

        for key in self.block.param_dict:
            self.assertEqual((answer[key] - self.block.param_dict[key]).simplify(), 0)

        assert (self.block.params_to_calibrate == [alpha])

        calibrating_eqs = [alpha - L / K + 0.36]

        for i, eq in enumerate(calibrating_eqs):
            self.assertEqual(eq.simplify(),
                             (self.block.params_to_calibrate[i] - self.block.calibrating_equations[i]).simplify())


if __name__ == '__main__':
    unittest.main()
