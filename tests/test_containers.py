import unittest
from typing import List

import sympy as sp
from sympy.polys.domains.mpelements import ComplexElement

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol


class TestSymbolDictErrors(unittest.TestCase):

    def test_raises_on_invalid_keys(self):
        with self.assertRaises(KeyError):
            # Only string or sp.Symbol keys allowed
            A = SymbolDictionary({1: 3})

    def test_raises_on_mixed_keys(self):
        with self.assertRaises(KeyError):
            # Cannot mix strings and symbols
            A = SymbolDictionary({'A': 3, sp.Symbol('B'): 2})

        with self.assertRaises(KeyError):
            B = SymbolDictionary({sp.Symbol('B'): 2, 'A' : 3})

    def test_setitem_raises_on_wrong_key_type(self):
        d = SymbolDictionary({sp.Symbol('A'): 3})
        with self.assertRaises(KeyError):
            d['B'] = 4

        d = SymbolDictionary({'A' : 3})
        with self.assertRaises(KeyError):
            d[sp.Symbol('B')] = 4

    def test_pipe_merge_errors_with_non_dict_other(self):
        d = SymbolDictionary({'A': 4})
        with self.assertRaises(ValueError) as e:
            s = {1, 2, 3}
            d | s
        error_msg = str(e.exception)
        self.assertEqual(error_msg, "__or__ not defined on non-dictionary objects")


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
        self.assertTrue(not d.is_sympy)

        self.d.to_string(inplace=True)
        self.assertEqual(list(self.d.keys()), ["C_t", "A_tp1", "r_tm1", "alpha"])

        self.assertTrue(not self.d.is_sympy)

    def test_convert_to_sympy(self):
        d = SymbolDictionary(dict(a=2, b=3)).to_sympy()
        self.assertEqual(list(d.keys()), [sp.Symbol("a"), sp.Symbol("b")])
        self.assertTrue(d.is_sympy)

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

    def test_step_forward_inplace(self):
        d2 = self.d.copy()
        d2.step_forward(inplace=True)
        keys = list(d2.to_string().keys())
        self.assertEqual(keys, ["C_tp1", "A_tp2", "r_t", "alpha"])

    def test_step_backward(self):
        d_tm1 = self.d.step_backward().to_string()

        keys = list(d_tm1.keys())
        self.assertEqual(keys, ["C_tm1", "A_t", "r_tm2", "alpha"])

    def test_step_backward_inplace(self):
        d2 = self.d.copy()
        d2.step_backward(inplace=True)
        keys = list(d2.to_string().keys())
        self.assertEqual(keys, ["C_tm1", "A_t", "r_tm2", "alpha"])

    def test_to_steady_state(self):
        d = self.d.to_ss().to_string()
        self.assertEqual(list(d.keys()), ["C_ss", "A_ss", "r_ss", "alpha"])

    def test_to_steady_state_inplace(self):
        d2 = self.d.copy()
        d2.to_ss(inplace=True)
        keys = list(d2.to_string().keys())
        self.assertEqual(keys, ["C_ss", "A_ss", "r_ss", "alpha"])

    def test_sort_dictionary(self):
        d = self.d.copy()

        d_sorted = d.sort_keys()
        self.assertEqual(list(d_sorted.keys()), [self.A, self.C, self.alpha, self.r])

        d.sort_keys(inplace=True)
        self.assertEqual(list(d.keys()), [self.A, self.C, self.alpha, self.r])

    def test_sequential_updates_from_empty(self):
        d0 = SymbolDictionary()
        d1 = SymbolDictionary({"A": 3, "B": 4})
        d2 = self.d.copy()

        self.assertRaises(ValueError, lambda: d1 | d2)

        def loop_update(ds: List):
            d0 = ds.pop(0)
            for d in ds:
                d0 = d0 | d
            return d0

        self.assertRaises(ValueError, loop_update, [d0, d1, d2])

    def test_convert_values(self):
        d = self.d.copy()
        d_sp = d.float_to_values()
        values = list(d_sp.values())
        self.assertTrue(all([isinstance(x, (sp.core.Number, ComplexElement)) for x in values]))

        d_np = d_sp.values_to_float()
        values = list(d_np.values())
        self.assertTrue(all([isinstance(x, (int, float, complex)) for x in values]))

    def test_convert_values_inplace(self):
        d = self.d.copy()
        d.float_to_values(inplace=True)
        values = list(d.values())
        self.assertTrue(all([isinstance(x, (sp.core.Number, ComplexElement)) for x in values]))

        d.values_to_float(inplace=True)
        values = list(d.values())
        self.assertTrue(all([isinstance(x, (int, float, complex)) for x in values]))

    def test_not_inplace_update_is_not_persistent(self):
        d = self.d
        d.to_string()

        self.assertTrue(all([isinstance(x, sp.Symbol) for x in d.keys()]))
        self.assertTrue(d.is_sympy)


if __name__ == "__main__":
    unittest.main()
