import unittest

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.utilities import (
    diff_through_time,
    step_equation_backward,
    step_equation_forward,
)


class TimeAwareSymbolTests(unittest.TestCase):
    def setUp(self):
        self.x_t = TimeAwareSymbol("x", 0)
        self.x_tp1 = TimeAwareSymbol("x", 1)
        self.x_tm1 = TimeAwareSymbol("x", -1)

    def test_base_name(self):
        self.assertEqual("x", self.x_t.base_name)

    def test_time_index(self):
        self.assertEqual(0, self.x_t.time_index)

    def test_step_forward(self):
        x_t1 = self.x_t.step_forward()
        self.assertEqual("x", x_t1.base_name)
        self.assertEqual("x_t+1", x_t1.name)
        self.assertEqual(1, x_t1.time_index)

    def test_step_backward(self):
        x_tm1 = self.x_t.step_backward()
        self.assertEqual("x", x_tm1.base_name)
        self.assertEqual("x_t-1", x_tm1.name)
        self.assertEqual(-1, x_tm1.time_index)

    def test_steady_state(self):
        x_ss = self.x_t.to_ss()
        self.assertEqual("x", x_ss.base_name)
        self.assertEqual("x_ss", x_ss.name)
        self.assertEqual("ss", x_ss.time_index)

    def test_equality_after_stepping(self):
        self.assertEqual(self.x_t.step_forward(), self.x_tp1)
        self.assertEqual(self.x_t.step_backward(), self.x_tm1)
        self.assertEqual(self.x_tp1.to_ss(), self.x_tm1.to_ss())

    def test_step_equation_backward(self):
        eq = self.x_t + self.x_tp1 + self.x_tm1
        self.assertEqual(step_equation_backward(eq), self.x_tm1 + self.x_t + TimeAwareSymbol("x", -2))

    def test_step_equation_forward(self):
        eq = self.x_t + self.x_tp1 + self.x_tm1
        self.assertEqual(step_equation_forward(eq), self.x_tp1 + TimeAwareSymbol("x", 2) + self.x_t)

    def test_diff_through_time(self):
        # X = Sum_{t=0}^10 beta * x_t
        # If we have 10 FoC for each x_t, then the Lagrangian is:
        # dL/dx_t = Sum_{t=0}^10 beta ** t

        X = sum(TimeAwareSymbol("x", t) for t in range(-10, 10))
        dX_dx_t = diff_through_time(X, self.x_t, sp.Symbol("beta"))

        self.assertEqual(dX_dx_t, sum(sp.Symbol("beta") ** t for t in range(11)))

    def test_diff_through_time_gap_in_time_indices(self):
        # Regression: when dx's base symbol appears at non-consecutive time indices, the derivative is zero at some
        # intermediate iterations but nonzero again later. The summation must walk through all relevant shifts, not exit
        # on the first zero. For eq = x[t] + a*x[t-2] differentiated wrt x[t]:
        #   k=0: contributes 1 (from x[t])
        #   k=1: contributes 0 (eq has x[t+1], x[t-1])
        #   k=2: contributes a*beta**2 (eq has x[t+2], x[t]; the t-2 term has shifted to t)
        a, beta = sp.symbols("a beta")
        x_tm2 = TimeAwareSymbol("x", -2)
        eq = self.x_t + a * x_tm2

        result = diff_through_time(eq, self.x_t, beta)
        expected = 1 + a * beta**2
        self.assertEqual(sp.simplify(result - expected), 0)

    def test_diff_through_time_no_appearance(self):
        # If dx's base symbol doesn't appear in eq, the result is zero — no iterations needed.
        y_t = TimeAwareSymbol("y", 0)
        eq = y_t + sp.Symbol("c")
        result = diff_through_time(eq, self.x_t, sp.Symbol("beta"))
        self.assertEqual(result, 0)

    def test_diff_through_time_only_forward_appearances(self):
        # If dx's base symbol appears only at time indices >= dx.time_index, only iter 0 can contribute (forward
        # stepping pushes everything further from dx). Confirm we don't iterate spuriously.
        x_tp3 = TimeAwareSymbol("x", 3)
        eq = self.x_t + sp.Symbol("a") * x_tp3
        result = diff_through_time(eq, self.x_t, sp.Symbol("beta"))
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
