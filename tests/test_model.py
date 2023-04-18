import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from numpy.testing import assert_allclose

from gEconpy.classes.model import gEconModel
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.parser.constants import DEFAULT_ASSUMPTIONS
from gEconpy.shared.utilities import string_keys_to_sympy

ROOT = Path(__file__).parent.absolute()


class ModelClassTestsOne(unittest.TestCase):
    def setUp(self):
        file_path = os.path.join(ROOT, "Test GCNs/One_Block_Simple_2.gcn")
        self.model = gEconModel(file_path, verbose=False)

    def test_model_options(self):
        self.assertEqual(self.model.options, {"output logfile": False, "output LaTeX": False})

    def test_reduce_vars_saved(self):
        self.assertEqual(
            self.model.try_reduce_vars,
            [TimeAwareSymbol("C", 0, **self.model.assumptions["C"])],
        )

    def test_model_file_loading(self):
        block_names = ["HOUSEHOLD"]
        result = [block_name for block_name in self.model.blocks.keys()]
        self.assertEqual(block_names, result)

        param_dict = {
            "theta": 0.357,
            "beta": 0.99,
            "delta": 0.02,
            "tau": 2,
            "rho": 0.95,
        }

        self.assertEqual(
            all([x in param_dict.keys() for x in self.model.free_param_dict.keys()]),
            True,
        )
        self.assertEqual(
            all([self.model.free_param_dict[x] == param_dict[x] for x in param_dict.keys()]),
            True,
        )
        self.assertEqual(
            self.model.params_to_calibrate,
            [sp.Symbol("alpha", **self.model.assumptions["alpha"])],
        )

    def test_conflicting_assumptions_are_removed(self):
        model = gEconModel(
            os.path.join(ROOT, "Test GCNs/conflicting_assumptions.gcn"), verbose=False
        )
        self.assertTrue("real" not in model.assumptions["TC"].keys())
        self.assertTrue("imaginary" in model.assumptions["TC"].keys())
        self.assertTrue(model.assumptions["TC"]["imaginary"])

    def test_solve_model_gensys(self):
        self.setUp()
        self.model.steady_state(verbose=False)
        self.assertEqual(self.model.steady_state_solved, True)
        self.model.solve_model(verbose=False, solver="gensys")
        self.assertEqual(self.model.perturbation_solved, True)

        # Values from R gEcon solution
        P = np.array([[0.950, 0.0000], [0.2710273, 0.8916969]])

        Q = np.array([[1.000], [0.2852917]])

        # TODO: Bug? When the SS value is negative, the sign of the S and R matrix entries are flipped relative to
        #   those of gEcon (row 4 -- Utility). This code flips the sign on my values to make the comparison.
        #   Check Dynare.
        R = np.array(
            [
                [0.70641931, 0.162459910],
                [13.55135517, -4.415155354],
                [0.42838971, -0.152667442],
                [-0.06008706, -0.009473984],
                [1.36634369, -0.072720705],
                [-0.80973441, -0.273514035],
                [-0.80973441, -0.273514035],
            ]
        )

        S = np.array(
            [
                [0.74359928],
                [14.26458439],
                [0.45093654],
                [-0.06324954],
                [1.43825652],
                [-0.85235201],
                [-0.85235201],
            ]
        )

        ss_df = pd.Series(string_keys_to_sympy(self.model.steady_state_dict))
        ss_df.index = list(map(lambda x: x.exit_ss().name, ss_df.index))
        # ss_df = ss_df.reindex(self.model.S.index)
        # neg_ss_mask = ss_df < 0

        A, _, _, _ = self.model.build_perturbation_matrices(
            np.fromiter(
                (self.model.free_param_dict | self.model.calib_param_dict).values(), dtype="float"
            ),
            np.fromiter(self.model.steady_state_dict.values(), dtype="float"),
        )

        (
            _,
            variables,
            _,
        ) = self.model.perturbation_solver.make_all_variable_time_combinations()

        gEcon_matrices = self.model.perturbation_solver.statespace_to_gEcon_representation(
            A, self.model.T.values, self.model.R.values, variables, 1e-7
        )
        model_P, model_Q, model_R, model_S, *_ = gEcon_matrices

        assert_allclose(model_P, P, equal_nan=True, err_msg="P", rtol=1e-5)
        assert_allclose(model_Q, Q, equal_nan=True, err_msg="Q", rtol=1e-5)
        assert_allclose(model_R, R, equal_nan=True, err_msg="R", rtol=1e-5)
        assert_allclose(model_S, S, equal_nan=True, err_msg="S", rtol=1e-5)

    def test_solve_model_cycle_reduction(self):
        self.setUp()
        self.model.steady_state(verbose=False)
        self.assertEqual(self.model.steady_state_solved, True)
        self.model.solve_model(verbose=False, solver="cycle_reduction")
        self.assertEqual(self.model.perturbation_solved, True)

        # Values from R gEcon solution
        P = np.array([[0.950, 0.0000], [0.2710273, 0.8916969]])

        Q = np.array([[1.000], [0.2852917]])

        # TODO: Check dynare outputs for sign flip
        R = np.array(
            [
                [0.70641931, 0.162459910],
                [13.55135517, -4.415155354],
                [0.42838971, -0.152667442],
                [-0.06008706, -0.009473984],
                [1.36634369, -0.072720705],
                [-0.80973441, -0.273514035],
                [-0.80973441, -0.273514035],
            ]
        )

        S = np.array(
            [
                [0.74359928],
                [14.26458439],
                [0.45093654],
                [-0.06324954],
                [1.43825652],
                [-0.85235201],
                [-0.85235201],
            ]
        )

        A, _, _, _ = self.model.build_perturbation_matrices(
            np.fromiter(
                (self.model.free_param_dict | self.model.calib_param_dict).values(), dtype="float"
            ),
            np.fromiter(self.model.steady_state_dict.values(), dtype="float"),
        )

        (
            _,
            variables,
            _,
        ) = self.model.perturbation_solver.make_all_variable_time_combinations()

        gEcon_matrices = self.model.perturbation_solver.statespace_to_gEcon_representation(
            A, self.model.T.values, self.model.R.values, variables, 1e-7
        )
        model_P, model_Q, model_R, model_S, *_ = gEcon_matrices

        self.assertEqual(np.allclose(model_P, P), True, msg="P")
        self.assertEqual(np.allclose(model_Q, Q), True, msg="Q")
        self.assertEqual(np.allclose(model_R, R), True, msg="R")
        self.assertEqual(np.allclose(model_S, S), True, msg="S")

    def test_solvers_agree(self):
        self.setUp()
        self.model.steady_state(verbose=False)
        self.model.solve_model(solver="gensys", verbose=False)
        Tg, Rg = self.model.T, self.model.R

        self.setUp()
        self.model.steady_state(verbose=False)
        self.model.solve_model(solver="cycle_reduction", verbose=False)
        Tc, Rc = self.model.T, self.model.R

        assert_allclose(
            Tg.round(5).values, Tc.round(5).values, rtol=1e-5, equal_nan=True, err_msg="T"
        )
        assert_allclose(
            Rg.round(5).values, Rc.round(5).values, rtol=1e-5, equal_nan=True, err_msg="R"
        )


class ModelClassTestsTwo(unittest.TestCase):
    def setUp(self):
        file_path = os.path.join(ROOT, "Test GCNs/Two_Block_RBC_1.gcn")
        self.model = gEconModel(file_path, verbose=False)

    def test_model_options(self):
        self.assertEqual(
            self.model.options,
            {
                "output logfile": True,
                "output LaTeX": True,
                "output LaTeX landscape": True,
            },
        )

    def test_reduce_vars_saved(self):
        self.assertEqual(self.model.try_reduce_vars, None)

    def test_model_file_loading(self):
        block_names = ["HOUSEHOLD", "FIRM"]
        result = [block_name for block_name in self.model.blocks.keys()]
        self.assertEqual(result, block_names)

        param_dict = {
            "beta": 0.985,
            "delta": 0.025,
            "sigma_C": 2,
            "sigma_L": 1.5,
            "alpha": 0.35,
            "rho_A": 0.95,
        }

        self.assertEqual(
            all([self.model.free_param_dict[x] == param_dict[x] for x in param_dict.keys()]),
            True,
        )
        self.assertEqual(self.model.params_to_calibrate, [])

    def test_solve_model_gensys(self):
        self.model.steady_state(verbose=False)
        self.assertEqual(self.model.steady_state_solved, True)
        self.model.solve_model(verbose=False, solver="gensys")
        self.assertEqual(self.model.perturbation_solved, True)

        P = np.array([[0.95000000, 0.0000000], [0.08887552, 0.9614003]])

        Q = np.array([[1.00000000], [0.09355318]])

        # TODO: Investigate sign flip on row 5, 6 (TC, U)
        R = np.array(
            [
                [0.3437521, 0.3981261],
                [3.5550207, -0.5439888],
                [0.1418896, -0.2412174],
                [1.0422283, 0.1932087],
                [-0.2127497, -0.1270917],
                [1.0422282, 0.1932087],
                [-0.6875042, -0.7962522],
                [-0.6875042, -0.7962522],
                [1.0422284, -0.8067914],
                [0.9003386, 0.4344261],
            ]
        )

        S = np.array(
            [
                [0.3618443],
                [3.7421271],
                [0.1493575],
                [1.0970824],
                [-0.2239471],
                [1.0970823],
                [-0.7236886],
                [-0.7236886],
                [1.0970825],
                [0.9477249],
            ]
        )

        A, _, _, _ = self.model.build_perturbation_matrices(
            np.fromiter(
                (self.model.free_param_dict | self.model.calib_param_dict).values(), dtype="float"
            ),
            np.fromiter(self.model.steady_state_dict.values(), dtype="float"),
        )

        (
            _,
            variables,
            _,
        ) = self.model.perturbation_solver.make_all_variable_time_combinations()

        gEcon_matrices = self.model.perturbation_solver.statespace_to_gEcon_representation(
            A, self.model.T.values, self.model.R.values, variables, 1e-7
        )
        model_P, model_Q, model_R, model_S, *_ = gEcon_matrices

        assert_allclose(model_P, P, equal_nan=True, err_msg="P", rtol=1e-5)
        assert_allclose(model_Q, Q, equal_nan=True, err_msg="Q", rtol=1e-5)
        assert_allclose(model_R, R, equal_nan=True, err_msg="R", rtol=1e-5)
        assert_allclose(model_S, S, equal_nan=True, err_msg="S", rtol=1e-5)

    def test_solve_model_cycle_reduction(self):
        self.model.steady_state(verbose=False)
        self.assertEqual(self.model.steady_state_solved, True)
        self.model.solve_model(verbose=False, solver="cycle_reduction")

        P = np.array([[0.95000000, 0.0000000], [0.08887552, 0.9614003]])

        Q = np.array([[1.00000000], [0.09355318]])

        # TODO: Investigate sign flip on row 5, 6 (TC, U)
        R = np.array(
            [
                [0.3437521, 0.3981261],
                [3.5550207, -0.5439888],
                [0.1418896, -0.2412174],
                [1.0422283, 0.1932087],
                [-0.2127497, -0.1270917],
                [1.0422282, 0.1932087],
                [-0.6875042, -0.7962522],
                [-0.6875042, -0.7962522],
                [1.0422284, -0.8067914],
                [0.9003386, 0.4344261],
            ]
        )

        S = np.array(
            [
                [0.3618443],
                [3.7421271],
                [0.1493575],
                [1.0970824],
                [-0.2239471],
                [1.0970823],
                [-0.7236886],
                [-0.7236886],
                [1.0970825],
                [0.9477249],
            ]
        )

        A, _, _, _ = self.model.build_perturbation_matrices(
            np.fromiter(
                (self.model.free_param_dict | self.model.calib_param_dict).values(), dtype="float"
            ),
            np.fromiter(self.model.steady_state_dict.values(), dtype="float"),
        )

        (
            _,
            variables,
            _,
        ) = self.model.perturbation_solver.make_all_variable_time_combinations()

        gEcon_matrices = self.model.perturbation_solver.statespace_to_gEcon_representation(
            A, self.model.T.values, self.model.R.values, variables, 1e-7
        )
        model_P, model_Q, model_R, model_S, *_ = gEcon_matrices

        assert_allclose(model_P, P, equal_nan=True, err_msg="P", rtol=1e-5)
        assert_allclose(model_Q, Q, equal_nan=True, err_msg="Q", rtol=1e-5)
        assert_allclose(model_R, R, equal_nan=True, err_msg="R", rtol=1e-5)
        assert_allclose(model_S, S, equal_nan=True, err_msg="S", rtol=1e-5)

    def test_solvers_agree(self):
        self.setUp()
        self.model.steady_state(verbose=False)
        self.model.solve_model(solver="gensys", verbose=False)
        Tg, Rg = self.model.T, self.model.R

        self.setUp()
        self.model.steady_state(verbose=False)
        self.model.solve_model(solver="cycle_reduction", verbose=False)
        Tc, Rc = self.model.T, self.model.R

        assert_allclose(
            Tg.round(5).values, Tc.round(5).values, rtol=1e-5, equal_nan=True, err_msg="T"
        )
        assert_allclose(
            Rg.round(5).values, Rc.round(5).values, rtol=1e-5, equal_nan=True, err_msg="R"
        )


class ModelClassTestsThree(unittest.TestCase):
    def setUp(self):
        file_path = os.path.join(ROOT, "Test GCNs/Full_New_Keyensian.gcn")
        self.model = gEconModel(
            file_path, verbose=False, simplify_constants=False, simplify_tryreduce=False
        )

    def test_model_options(self):
        self.assertEqual(
            self.model.options,
            {
                "output logfile": True,
                "output LaTeX": True,
                "output LaTeX landscape": True,
            },
        )

    def test_reduce_vars_saved(self):
        self.assertEqual(
            self.model.try_reduce_vars,
            [
                "Div[]",
                "TC[]"
                # TimeAwareSymbol("Div", 0, **self.model.assumptions["DIV"]),
                # TimeAwareSymbol("TC", 0, **self.model.assumptions["TC"]),
            ],
        )

    def test_model_file_loading(self):
        block_names = [
            "HOUSEHOLD",
            "WAGE_SETTING",
            "WAGE_EVOLUTION",
            "PREFERENCE_SHOCKS",
            "FIRM",
            "TECHNOLOGY_SHOCKS",
            "FIRM_PRICE_SETTING_PROBLEM",
            "PRICE_EVOLUTION",
            "MONETARY_POLICY",
            "EQUILIBRIUM",
        ]

        result = [block_name for block_name in self.model.blocks.keys()]
        self.assertEqual(result, block_names)

        (rho_technology, gamma_R, gamma_pi, gamma_Y, phi_pi_obj, phi_pi, rho_pi_dot,) = sp.symbols(
            [
                "rho_technology",
                "gamma_R",
                "gamma_pi",
                "gamma_Y",
                "phi_pi_obj",
                "phi_pi",
                "rho_pi_dot",
            ],
            **DEFAULT_ASSUMPTIONS
        )

        param_dict = {
            "delta": 0.025,
            "beta": 0.99,
            "sigma_C": 2,
            "sigma_L": 1.5,
            "gamma_I": 10,
            "phi_H": 0.5,
            "psi_w": 0.782,
            "eta_w": 0.75,
            "alpha": 0.35,
            "rho_technology": 0.95,
            "rho_preference": 0.95,
            "psi_p": 0.6,
            "eta_p": 0.75,
            "gamma_R": 0.9,
            "gamma_pi": 1.5,
            "gamma_Y": 0.05,
            "rho_pi_dot": 0.924,
        }

        self.assertEqual(
            all([x in param_dict.keys() for x in self.model.free_param_dict.keys()]),
            True,
        )
        self.assertEqual(
            all([self.model.free_param_dict[x] == param_dict[x] for x in param_dict.keys()]),
            True,
        )
        self.assertEqual(self.model.params_to_calibrate, [phi_pi, phi_pi_obj])

    # def test_solvers_agree(self):
    #     self.setUp()
    #     self.model.steady_state(verbose=False)
    #     self.model.solve_model(solver="gensys", verbose=True)
    #     Tg, Rg = self.model.T, self.model.R
    #
    #     self.setUp()
    #     self.model.steady_state(verbose=False)
    #     self.model.solve_model(solver="cycle_reduction", verbose=True)
    #     Tc, Rc = self.model.T, self.model.R
    #
    #     assert_allclose(Tg.values, Tc.values, rtol=1e-5, equal_nan=True, err_msg='T')
    #     assert_allclose(Rg.values, Rc.values, rtol=1e-5, equal_nan=True, err_msg='R')

    def test_solvers_agree(self):
        self.model.steady_state(verbose=False)
        self.model.solve_model(solver="gensys", verbose=False)
        Tg, Rg = self.model.T, self.model.R

        self.model.solve_model(solver="cycle_reduction", verbose=False)
        Tc, Rc = self.model.T, self.model.R

        assert_allclose(Tg.values, Tc.values, rtol=1e-5, equal_nan=True, err_msg="T")
        assert_allclose(Rg.values, Rc.values, rtol=1e-5, equal_nan=True, err_msg="R")

    # def test_solve_model(self):
    #     self.model.steady_state(verbose=False)

    #     self.model.solve_model(verbose=False, solver='gensys')
    #
    #     P = np.array([[0.92400000, 0.00000000, 0.000000000, 0.000000000, 0.000000000, 0.0000000000, 0.0000000000,
    #                    0.000000000, 0.00000000, 0.0000000000],
    #                   [0.04464553, 0.77386407, 0.008429303, -0.035640523, 0.019260369, -0.0061647545, 0.0064098938,
    #                    0.003811426, -0.01635691, -0.0042992448],
    #                   [0.00000000, 0.00000000, 0.950000000, 0.000000000, 0.000000000, 0.0000000000, 0.0000000000,
    #                    0.000000000, 0.00000000, 0.0000000000],
    #                   [0.00000000, 0.00000000, 0.000000000, 0.950000000, 0.000000000, 0.0000000000, 0.0000000000,
    #                    0.000000000, 0.00000000, 0.0000000000],
    #                   [0.11400712, -0.23033661, 0.017018503, 0.246571939, 0.714089188, 0.0015115630, -0.0025199985,
    #                    0.003439315, 0.09953510, 0.0012796478],
    #                   [0.00000000, 0.00000000, 0.000000000, 0.000000000, 0.000000000, 0.0000000000, 0.0000000000,
    #                    0.000000000, 0.00000000, 0.0000000000],
    #                   [0.56944713, -1.31534877, 0.116205871, 0.279528217, -0.069058930, 0.0055509980, 0.4892113664,
    #                    0.001268753, 0.13710342, 0.0073074932],
    #                   [0.77344786, -1.65448037, -0.084222852, 0.373554371, -0.110359402, 0.0067463467, -0.0129713461,
    #                    0.893824526, -0.07071734, 0.0091915576],
    #                   [0.01933620, -0.04136201, -0.002105571, 0.009338859, -0.002758985, 0.0001686587, -0.0003242837,
    #                    0.022345613, 0.97323207, 0.0002297889],
    #                   [0.60123052, -1.36818560, 0.084979004, 0.294177526, -0.075493558, -0.5547430352, 0.4109711206,
    #                    0.140329261, 0.10472487, 0.0076010311]])
    #
    #     Q = np.array([[0.000000000, 0.000000000, 0.00000000, 1.00000000],
    #                   [0.008872950, -0.037516340, 0.85984896, 0.04831767],
    #                   [1.000000000, 0.000000000, 0.00000000, 0.00000000],
    #                   [0.000000000, 1.000000000, 0.00000000, 0.00000000],
    #                   [0.017914213, 0.259549409, -0.25592956, 0.12338433],
    #                   [0.000000000, 0.000000000, 0.00000000, 0.00000000],
    #                   [0.122321970, 0.294240229, -1.46149864, 0.61628477],
    #                   [-0.088655634, 0.393215127, -1.83831153, 0.83706479],
    #                   [-0.002216391, 0.009830378, -0.04595779, 0.02092662],
    #                   [0.089451584, 0.309660553, -1.52020622, 0.65068238]])
    #
    #     R = np.array([[-2.70120790, 6.4759672, 0.45684368, -1.0523862, 0.25304694, -0.028589270, 0.043922008,
    #                    -0.010211851, -0.50854833, -0.0359775957],
    #                   [0.43774664, -0.9670519, 0.06277643, -1.0565632, 0.67343881, -0.297196226, 0.218772144,
    #                    0.079001225, -0.38253612, 0.0053725107],
    #                   [0.58559582, -0.7953000, 0.05336272, -0.2474094, 0.13091891, -0.022606929, 0.029033588,
    #                    0.020731865, -0.11253692, 0.0044183336],
    #                   [1.75678747, -2.3859001, 0.16008816, -0.7422282, 0.39275674, -0.067820786, 0.087100765,
    #                    0.062195594, -0.33761076, 0.0132550007],
    #                   [-0.34114299, 0.5424464, 0.48057739, -0.7361740, 0.12517618, -0.002047156, 0.028009363,
    #                    -0.063210365, -0.75978505, -0.0030135913],
    #                   [1.03897717, -2.3352376, 0.14775544, -0.7623857, 0.59794526, -0.851939276, 0.629743275,
    #                    0.219330490, -1.27781127, 0.0129735420],
    #                   [2.21281597, -3.3072465, 0.22816217, 0.2440595, 0.24911350, -0.061774534, 0.077020771,
    #                    0.075952852, 0.06052965, 0.0183735919],
    #                   [0.92497003, -2.1049009, 0.13073693, -1.0089577, -0.11614394, -0.853450826, 0.632263264,
    #                    0.215891172, -0.37734635, 0.0116938940],
    #                   [-1.86247082, 3.7798186, 0.45779728, -1.0016774, 0.69227986, -0.140940083, 0.177078457,
    #                    0.079706624, -0.54739326, -0.0209989924],
    #                   [2.76788546, -2.2659060, 0.88668501, -1.6781507, 0.64733010, -0.213012025, 0.289374259,
    #                    0.186334186, -0.95855542, 0.0125883667],
    #                   [-1.86247082, 3.7798186, 0.45779728, -1.0016774, 0.69227986, -0.140940083, 0.177078457,
    #                    0.079706624, -0.54739326, -0.0209989924],
    #                   [2.76788546, -2.2659060, 0.88668501, -1.6781507, 0.64733010, -0.213012025, 0.289374259,
    #                    0.186334186, -0.95855542, 0.0125883667],
    #                   [0.07745758, -0.1102706, -0.16967797, 0.1782883, -0.01302221, 0.002553406, -0.004433502,
    #                    0.007300968, 0.07817343, 0.0006126142]])
    #
    #     S = np.array([[0.48088808, -1.1077749, 7.1955191, -2.92338518],
    #                   [0.06608045, -1.1121718, -1.0745021, 0.47375177],
    #                   [0.05617128, -0.2604309, -0.8836667, 0.63376171],
    #                   [0.16851385, -0.7812928, -2.6510001, 1.90128514],
    #                   [0.50587094, -0.7749200, 0.6027183, -0.36920237],
    #                   [0.15553204, -0.8025113, -2.5947084, 1.12443417],
    #                   [0.24017070, 0.2569048, -3.6747184, 2.39482247],
    #                   [0.13761782, -1.0620607, -2.3387788, 1.00104982],
    #                   [0.48189187, -1.0543973, 4.1997985, -2.01566106],
    #                   [0.93335265, -1.7664744, -2.5176733, 2.99554704],
    #                   [0.48189187, -1.0543973, 4.1997985, -2.01566106],
    #                   [0.93335265, -1.7664744, -2.5176733, 2.99554704],
    #                   [-0.17860839, 0.1876719, -0.1225228, 0.08382855]])
    #
    #     index_10 = ['pi_obj', 'r_G', 'shock_preference', 'shock_technology', 'w', 'B', 'C', 'I', 'K', 'Y']
    #     cols_10 = ['pi_obj', 'r_G', 'shock_preference', 'shock_technology', 'w', 'B', 'C',  'I', 'K', 'Y']
    #
    #
    #
    #     index_11 = ['lambda_t', 'q_t', 'r_t', 'w_t', 'C_t', 'I_t', 'L_t', 'P_t', 'TC_t', 'U_t', 'Y_t']
    #     ss_df = pd.Series(self.model.steady_state_dict)
    #     ss_df.index = list(map(lambda x: x.exit_ss().name, ss_df.index))
    #     ss_df = ss_df.reindex(self.model.S.index)
    #     neg_ss_mask = ss_df < 0
    #
    #     for answer, result in zip([P, Q, R, S], [self.model.P, self.model.Q, self.model.R, self.model.S]):
    #         if result.shape[0] == 11:
    #             result = result.loc[index_11, :]
    #             result.loc[neg_ss_mask, :] = result.loc[neg_ss_mask, :] * -1
    #         self.assertEqual(np.allclose(answer, result.values), True)


if __name__ == "__main__":
    unittest.main()
