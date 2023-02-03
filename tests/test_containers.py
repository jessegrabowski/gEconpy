import unittest

import sympy as sp

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol


class TestSymbolDictionary(unittest.TestCase):
    def setUp(self) -> None:
        self.C = C = TimeAwareSymbol("C", 0, positive=True)
        self.A = A = TimeAwareSymbol("A", 1, negative=True)
        self.r = r = TimeAwareSymbol("r", -1, imaginary=True)
        self.alpha = alpha = sp.Symbol("alpha", real=True)

        self.d = SymbolDictionary({C: 1, A: -1, r: 2j, alpha: 0.3})

    def test_convert_to_string(self):
        d = self.d.to_string()
        self.assertEqual(list(d.keys()), ["C_t", "A_tp1", "r_tm1", "alpha"])

        self.d.to_string(inplace=True)
        self.assertEqual(list(self.d.keys()), ["C_t", "A_tp1", "r_tm1", "alpha"])

    def test_convert_to_sympy(self):
        d = SymbolDictionary(dict(a=2, b=3)).to_sympy()
        self.assertEqual(list(d.keys()), [sp.Symbol("a"), sp.Symbol("b")])

    def test_copy(self):
        d_copy = self.d.copy()
        d_ref = self.d

        self.assertTrue(self.d is d_ref)
        self.assertTrue(self.d is not d_copy)

    def test_assumptions_preserved(self):
        assumptions = self.d._assumptions.copy()
        d = self.d.to_string()

        self.assertEqual(d._assumptions, assumptions)

        d = self.d.to_sympy()
        self.assertEqual(d._assumptions, assumptions)

        d = self.d.values_to_float()
        self.assertEqual(d._assumptions, assumptions)

    def test_join_with_pipe(self):
        F = TimeAwareSymbol("F", "ss")
        d1 = self.d.copy()
        d2 = SymbolDictionary({F: 3})

        new_d = self.d | d2
        new_d.sort_keys(inplace=True)

        self.assertEqual(list(new_d.keys()), [self.A, self.C, F, self.alpha, self.r])
        self.assertEqual(self.d._assumptions, d1._assumptions | d2._assumptions)

    def test_step_forward(self):
        d_tp1 = self.d.step_forward().to_string()

        keys = list(d_tp1.keys())
        self.assertEqual(keys, ["C_tp1", "A_tp2", "r_t", "alpha"])

    def test_step_backward(self):
        d_tm1 = self.d.step_backward().to_string()

        keys = list(d_tm1.keys())
        self.assertEqual(keys, ["C_tm1", "A_t", "r_tm2", "alpha"])

    def test_to_steady_state(self):
        d = self.d.to_ss().to_string()
        self.assertEqual(list(d.keys()), ["C_ss", "A_ss", "r_ss", "alpha"])

    def test_sort_dictionary(self):
        d = self.d.copy()

        d_sorted = d.sort_keys()
        self.assertEqual(list(d_sorted.keys()), [self.A, self.C, self.alpha, self.r])

        d.sort_keys(inplace=True)
        self.assertEqual(list(d.keys()), [self.A, self.C, self.alpha, self.r])


if __name__ == "__main__":
    unittest.main()
