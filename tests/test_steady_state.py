import os
import unittest
from pathlib import Path

import sympy as sp
from scipy import optimize

from gEconpy.classes.model import gEconModel

ROOT = Path(__file__).parent.absolute()


class SteadyStateModelOne(unittest.TestCase):
    def setUp(self):
        self.model = gEconModel(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn"), verbose=False
        )

    def test_solve_ss_with_partial_user_solution(self):
        self.model = gEconModel(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn"), verbose=False
        )
        self.model.steady_state(verbose=False, apply_user_simplifications=True)
        self.assertTrue(self.model.steady_state_solved)

    def test_wrong_user_solutions_raises(self):
        self.model = gEconModel(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn"), verbose=False
        )
        self.model.steady_state_relationships["A_ss"] = 3.0

        self.assertRaises(ValueError, self.model.steady_state, verbose=False)

    def test_incomplete_ss_relationship_raises_with_root(self):
        self.model = gEconModel(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn"), verbose=False
        )
        self.model.steady_state_relationships["K_ss"] = 3.0
        self.assertRaises(ValueError, self.model.steady_state, verbose=False, method="root")

    def test_wrong_and_incomplete_ss_relationship_fails_with_minimize(self):
        self.model = gEconModel(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn"), verbose=False
        )
        self.model.steady_state_relationships["K_ss"] = 3.0
        self.model.steady_state(method="minimize", verbose=False)

        self.assertTrue(not self.model.steady_state_solved)

    def test_numerical_solvers_suceed_and_agree(self):
        self.model = gEconModel(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_1.gcn"), verbose=False
        )
        self.model.steady_state(method="root", verbose=False)
        self.assertTrue(self.model.steady_state_solved)
        ss_root = self.model.steady_state_dict.copy()

        self.model.steady_state(method="minimize", verbose=False)
        self.assertTrue(self.model.steady_state_solved)
        ss_minimize = self.model.steady_state_dict.copy()

        for k in ss_root.keys():
            self.assertAlmostEqual(ss_root[k], ss_minimize[k], places=6, msg=k)

    def test_steady_state_matches_analytic(self):
        param_dict = self.model.free_param_dict.to_sympy()
        alpha, beta, delta, gamma, rho = list(param_dict.keys())

        A_ss = sp.Float(1.0)
        K_ss = ((alpha * beta) / (1 - beta + beta * delta)) ** (1 / (1 - alpha))
        C_ss = K_ss**alpha - delta * K_ss
        lambda_ss = C_ss ** (-gamma)
        U_ss = 1 / (1 - beta) * (C_ss ** (1 - gamma) - 1) / (1 - gamma)

        ss_var = [x.to_ss() for x in self.model.variables]
        ss_dict = {
            k: v.subs(param_dict) for k, v in zip(ss_var, [A_ss, C_ss, K_ss, U_ss, lambda_ss])
        }

        self.model.steady_state(verbose=False)
        self.assertTrue("A_ss" in self.model.steady_state_dict.keys())

        for k in ss_dict:
            self.assertAlmostEqual(ss_dict[k], self.model.steady_state_dict[k.name], places=5)


class SteadyStateModelTwo(unittest.TestCase):
    def setUp(self):
        self.model = gEconModel(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_2.gcn"), verbose=False
        )

    def test_numerical_solvers_succeed_and_agree(self):
        self.model = gEconModel(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_2.gcn"), verbose=False
        )
        self.model.steady_state(method="root", verbose=False)
        self.assertTrue(self.model.steady_state_solved)
        ss_root = self.model.steady_state_dict.copy()

        self.model.steady_state(method="minimize", verbose=False)
        self.assertTrue(self.model.steady_state_solved)
        ss_minimize = self.model.steady_state_dict.copy()

        for k in ss_root.keys():
            self.assertAlmostEqual(ss_root[k], ss_minimize[k], places=6, msg=k)

    def test_steady_state_matches_analytic(self):
        param_dict = self.model.free_param_dict.to_sympy()
        calib_params = self.model.params_to_calibrate

        beta, delta, rho, tau, theta = list(param_dict.keys())
        (alpha,) = calib_params

        term_1 = theta * (1 - alpha) / (1 - theta)
        term_2 = alpha / (1 - beta + beta * delta)
        a_exp = alpha / (1 - alpha)

        A_ss = sp.Float(1.0)
        Y_ss = term_1 * term_2**a_exp / (1 + term_1 - delta * term_2)
        K_ss = term_2 * Y_ss
        L_ss = term_2 ** (-a_exp) * Y_ss
        C_ss = term_1 * term_2**a_exp - term_1 * Y_ss
        I_ss = delta * K_ss

        lambda_ss = theta * (C_ss**theta * (1 - L_ss) ** (1 - theta)) ** (1 - tau) / C_ss

        U_ss = 1 / (1 - beta) * (C_ss**theta * (1 - L_ss) ** (1 - theta)) ** (1 - tau) / (1 - tau)
        f = sp.lambdify(alpha, (L_ss / K_ss - 0.36).simplify().subs(param_dict))

        res = optimize.root_scalar(f, bracket=[1e-4, 0.99])
        calib_solution = {alpha: res.root}
        all_params = param_dict | calib_solution

        ss_var = [x.to_ss() for x in self.model.variables]
        A, C, I, K, L, U, Y, lam, q = ss_var
        ss_dict = {
            k: v.subs(all_params)
            for k, v in zip(
                ss_var, [A_ss, C_ss, I_ss, K_ss, L_ss, U_ss, Y_ss, lambda_ss, lambda_ss]
            )
        }

        self.assertAlmostEqual(ss_dict[L] / ss_dict[K], 0.36)

        self.model.steady_state(verbose=False)

        for k in ss_dict:
            self.assertAlmostEqual(ss_dict[k], self.model.steady_state_dict[k.name], places=5)


class SteadyStateModelThree(unittest.TestCase):
    def setUp(self):
        self.model = gEconModel(os.path.join(ROOT, "Test GCNs/Two_Block_RBC_1.gcn"), verbose=False)
        self.model.steady_state(verbose=False)

    def test_numerical_solvers_succeed_and_agree(self):
        self.model = gEconModel(os.path.join(ROOT, "Test GCNs/Two_Block_RBC_1.gcn"), verbose=False)
        self.model.steady_state(method="root", verbose=False)
        self.assertTrue(self.model.steady_state_solved)
        ss_root = self.model.steady_state_dict.copy()

        self.model.steady_state(method="minimize", verbose=False)
        self.assertTrue(self.model.steady_state_solved)
        ss_minimize = self.model.steady_state_dict.copy()

        for k in ss_root.keys():
            self.assertAlmostEqual(ss_root[k], ss_minimize[k], places=6, msg=k)

    def test_steady_state_matches_analytic(self):
        param_dict = self.model.free_param_dict.to_sympy()

        alpha, beta, delta, rho_A, sigma_C, sigma_L = list(param_dict.keys())
        A_ss = sp.Float(1.0)
        r_ss = 1 / beta - (1 - delta)
        w_ss = (1 - alpha) * (alpha / r_ss) ** (alpha / (1 - alpha))
        Y_ss = (
            w_ss ** (1 / (sigma_L + sigma_C))
            * (w_ss / (1 - alpha)) ** (sigma_L / (sigma_L + sigma_C))
            * (r_ss / (r_ss - delta * alpha)) ** (sigma_C / (sigma_L + sigma_C))
        )

        C_ss = (w_ss) ** (1 / sigma_C) * (w_ss / (1 - alpha) / Y_ss) ** (sigma_L / sigma_C)

        lambda_ss = C_ss ** (-sigma_C)
        q_ss = lambda_ss
        I_ss = delta * alpha * Y_ss / r_ss
        K_ss = alpha * Y_ss / r_ss
        L_ss = (1 - alpha) * Y_ss / w_ss
        P_ss = (w_ss / (1 - alpha)) ** (1 - alpha) * (r_ss / alpha) ** alpha

        U_ss = (
            1
            / (1 - beta)
            * (C_ss ** (1 - sigma_C) / (1 - sigma_C) - L_ss ** (1 + sigma_L) / (1 + sigma_L))
        )

        TC_ss = -(r_ss * K_ss + w_ss * L_ss)

        ss_var = [x.to_ss() for x in self.model.variables]
        answers = [
            A_ss,
            C_ss,
            I_ss,
            K_ss,
            L_ss,
            TC_ss,
            U_ss,
            Y_ss,
            lambda_ss,
            q_ss,
            r_ss,
            w_ss,
        ]
        ss_dict = {k: v.subs(param_dict) for k, v in zip(ss_var, answers)}

        for k in ss_dict:
            self.assertAlmostEqual(ss_dict[k], self.model.steady_state_dict[k.name], places=5)

        self.assertAlmostEqual(P_ss.subs(param_dict), 1.0)


class SteadyStateModelFour(unittest.TestCase):
    def setUp(self):
        self.model = gEconModel(
            os.path.join(ROOT, "Test GCNs/Full_New_Keyensian.gcn"), verbose=False
        )
        self.model.steady_state(verbose=False)

    def test_numerical_solvers_succeed_and_agree(self):
        self.model = gEconModel(
            os.path.join(ROOT, "Test GCNs/Full_New_Keyensian.gcn"), verbose=False
        )
        self.model.steady_state(method="root", verbose=False)
        self.assertTrue(self.model.steady_state_solved)
        ss_root = self.model.steady_state_dict.copy()

        self.model.steady_state(method="minimize", verbose=False)
        self.assertTrue(self.model.steady_state_solved)
        ss_minimize = self.model.steady_state_dict.copy()

        for k in ss_root.keys():
            self.assertAlmostEqual(ss_root[k], ss_minimize[k], places=6, msg=k)

    def test_steady_state_matches_analytic(self):
        param_dict = self.model.free_param_dict.to_sympy()
        (
            alpha,
            beta,
            delta,
            eta_p,
            eta_w,
            gamma_I,
            gamma_R,
            gamma_Y,
            gamma_pi,
            phi_H,
            psi_p,
            psi_w,
            rho_pi_dot,
            rho_preference,
            rho_technology,
            sigma_C,
            sigma_L,
        ) = list(param_dict.keys())

        shock_technology_ss = sp.Float(1)
        shock_preference_ss = sp.Float(1)
        pi_ss = sp.Float(1)
        pi_star_ss = sp.Float(1)
        pi_obj_ss = sp.Float(1)
        B_ss = sp.Float(0)

        r_ss = 1 / beta - (1 - delta)
        r_G_ss = 1 / beta

        mc_ss = 1 / (1 + psi_p)
        w_ss = (1 - alpha) * mc_ss ** (1 / (1 - alpha)) * (alpha / r_ss) ** (alpha / (1 - alpha))

        w_star_ss = w_ss

        Y_ss = (
            w_ss ** ((sigma_L + 1) / (sigma_C + sigma_L))
            * ((-beta * phi_H + 1) / (psi_w + 1)) ** (1 / (sigma_C + sigma_L))
            * (r_ss / ((1 - phi_H) * (-alpha * delta * mc_ss + r_ss)))
            ** (sigma_C / (sigma_C + sigma_L))
            / (mc_ss * (1 - alpha)) ** (sigma_L / (sigma_C + sigma_L))
        )

        C_ss = (
            w_ss ** ((1 + sigma_L) / sigma_C)
            * (1 / (1 - phi_H))
            * ((1 - beta * phi_H) / (1 + psi_w)) ** (1 / sigma_C)
            * ((1 - alpha) * mc_ss) ** (-sigma_L / sigma_C)
            * Y_ss ** (-sigma_L / sigma_C)
        )

        lambda_ss = (1 - beta * phi_H) * ((1 - phi_H) * C_ss) ** (-sigma_C)
        q_ss = lambda_ss
        I_ss = delta * alpha * mc_ss * Y_ss / r_ss
        K_ss = alpha * mc_ss * Y_ss / r_ss
        L_ss = (1 - alpha) * Y_ss * mc_ss / w_ss

        U_ss = (
            1
            / (1 - beta)
            * (
                ((1 - phi_H) * C_ss) ** (1 - sigma_C) / (1 - sigma_C)
                - L_ss ** (1 + sigma_L) / (1 + sigma_L)
            )
        )

        TC_ss = -(r_ss * K_ss + w_ss * L_ss)
        Div_ss = Y_ss + TC_ss

        LHS_ss = 1 / (1 - beta * eta_p * pi_ss ** (1 / psi_p)) * lambda_ss * Y_ss * pi_star_ss

        RHS_ss = 1 / (1 + psi_p) * LHS_ss

        LHS_w_ss = 1 / (1 - beta * eta_w) * 1 / (1 + psi_w) * w_star_ss * lambda_ss * L_ss

        RHS_w_ss = LHS_w_ss

        ss_var = [x.to_ss() for x in self.model.variables]
        answers = [
            C_ss,
            Div_ss,
            I_ss,
            K_ss,
            LHS_ss,
            LHS_w_ss,
            L_ss,
            RHS_ss,
            RHS_w_ss,
            TC_ss,
            U_ss,
            Y_ss,
            lambda_ss,
            mc_ss,
            pi_obj_ss,
            pi_star_ss,
            pi_ss,
            q_ss,
            r_G_ss,
            r_ss,
            shock_preference_ss,
            shock_technology_ss,
            w_star_ss,
            w_ss,
        ]

        ss_dict = {k: v.subs(param_dict) for k, v in zip(ss_var, answers)}

        for k in ss_dict:
            self.assertAlmostEqual(
                ss_dict[k], self.model.steady_state_dict[k.name], places=5, msg=k
            )


class SteadyStateWithUserError(unittest.TestCase):
    def setUp(self):
        self.model = gEconModel(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_1_ss_Error.gcn"), verbose=False
        )

    def test_raises_on_nonzero_resids(self):
        self.assertRaises(
            ValueError, self.model.steady_state, apply_user_simplifications=True, verbose=False
        )


class FullyUserDefinedSteadyState(unittest.TestCase):
    def test_ss_solves_from_user_definition(self):
        model = gEconModel(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_1_w_Steady_State.gcn"), verbose=False
        )

        for method in ["root", "minimize"]:
            model.steady_state(apply_user_simplifications=True, verbose=False, method=method)
            self.assertTrue(model.steady_state_solved, msg=method)

    def test_ss_solves_when_ignoring_user_definition(self):
        model = gEconModel(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_1_w_Steady_State.gcn"), verbose=False
        )

        for method in ["root", "minimize"]:
            model.steady_state(apply_user_simplifications=False, verbose=False, method=method)
            self.assertTrue(model.steady_state_solved, msg=method)

    def test_solver_matches_user_solution(self):
        model = gEconModel(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_1_w_Steady_State.gcn"), verbose=False
        )
        model.steady_state(apply_user_simplifications=False, verbose=False)
        ss_dict_numeric = model.steady_state_dict.copy()

        model = gEconModel(
            os.path.join(ROOT, "Test GCNs/One_Block_Simple_1_w_Steady_State.gcn"), verbose=False
        )
        model.steady_state(apply_user_simplifications=True, verbose=False)
        ss_dict_user = model.steady_state_dict.copy()

        for k in ss_dict_user:
            self.assertAlmostEqual(ss_dict_numeric[k], ss_dict_user[k], msg=k)


class TestRBCComplete(unittest.TestCase):
    def test_steady_state(self):
        mod = gEconModel("../GCN Files/RBC_complete.gcn", verbose=False)
        mod.steady_state(verbose=False)
        self.assertTrue(mod.steady_state_solved)


if __name__ == "__main__":
    unittest.main()
