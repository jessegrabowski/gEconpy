import unittest
from gEcon.classes.model import gEconModel
from gEcon.solvers.steady_state import SteadyStateSolver
import sympy as sp
import numpy as np
from sympy.abc import x,y,z,a,b,c
import os


class SteadyStateSolverTests(unittest.TestCase):

    def setUp(self):
        model = gEconModel('Test GCNs/One_Block_Simple_1.gcn', verbose=False)
        self.solver = SteadyStateSolver(model)

    def test_create_jacobian(self):
        equations = [a + x**2, a*x*y]
        f_jac = self.solver._build_jacobian(diff_variables=['x', 'y'],
                                            additional_inputs=['a'],
                                            equations=equations)
        expected_result = np.array([[4.0, 0.0],
                                    [1.0, 2.0]])

        self.assertTrue(np.allclose(f_jac(args=[], kwargs={'a':1.0, 'x':2.0, 'y':1.0}), expected_result))


class NoUserInfoSteadyStateTest(unittest.TestCase):

    def setUp(self):
        self.model = gEconModel('Test GCNs/One_Block_Simple_1.gcn', verbose=False)

    def test_steady_state(self):
        self.model.steady_state(verbose=False)
        self.assertTrue(self.model.steady_state_solved)


class FullModelSolutionProvidedTest(unittest.TestCase):
    def setUp(self):
        self.model = gEconModel('Test GCNs/One_Block_Simple_1_w_Steady_State.gcn', verbose=False)

    def test_steady_state(self):
        self.model.steady_state(verbose=False)
        self.assertTrue(self.model.steady_state_solved)


class BigModelSteadyStateTest(unittest.TestCase):

    def setUp(self):
        self.model = gEconModel('Test GCNs/Full_New_Keyensian.gcn', verbose=False)

    def test_something(self):
        self.model.steady_state(verbose=False)
        self.assertTrue(self.model.steady_state_solved)


if __name__ == '__main__':
    unittest.main()
