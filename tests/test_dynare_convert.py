import os
import unittest

from pathlib import Path

import numpy as np
import sympy as sp

from gEconpy import model_from_gcn
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.shared.dynare_convert import (
    build_hash_table,
    convert_var_timings_to_matlab,
    get_name,
    make_mod_file,
    make_var_to_matlab_sub_dict,
    substitute_equation_from_dict,
    write_lines_from_list,
)

ROOT = Path(__file__).parent.absolute()


class TestDynareConvert(unittest.TestCase):
    def test_get_name(self):
        cases = [sp.Symbol("x"), TimeAwareSymbol("y", 1), "test"]
        responses = ["x", "y_tp1", "test"]
        for case, answer in zip(cases, responses):
            self.assertEqual(answer, get_name(case))

    def test_build_hash_table_from_strings(self):
        test_list = ["4", "*", "y", "+", "x", "=", "-4"]
        var_to_hash, hash_to_var = build_hash_table(test_list)
        self.assertTrue(all([x in var_to_hash for x in test_list]))

        hashed_string = "_".join([var_to_hash.get(x) for x in test_list])
        unhashed_string = "_".join(
            [hash_to_var.get(x) for x in hashed_string.split("_")]
        )

        self.assertEqual("_".join(test_list), unhashed_string)

    def test_build_hash_table_from_symbols(self):
        test_list = ["4", "*", sp.Symbol("y"), "+", sp.Symbol("x"), "=", "-4"]
        var_to_hash, hash_to_var = build_hash_table(test_list)
        self.assertTrue(all([get_name(x) in var_to_hash for x in test_list]))

        hashed_list = [var_to_hash.get(get_name(x)) for x in test_list]
        unhashed_list = [hash_to_var.get(x) for x in hashed_list]

        self.assertEqual([get_name(x) for x in test_list], unhashed_list)

    def test_replace_equations_with_hashes(self):
        eq_str = "2 * y + 3 * x ^ 2 = -23"
        tokens = ["y", "x"]
        var_to_hash, hash_to_var = build_hash_table(tokens)

        hashed_eq = substitute_equation_from_dict(eq_str, var_to_hash)
        unhashed_eq = substitute_equation_from_dict(hashed_eq, hash_to_var)

        self.assertEqual(unhashed_eq, eq_str)

    def test_make_var_to_matlab_sub_dict(self):
        variables = [sp.Symbol("beta"), TimeAwareSymbol("gamma", 0), "lambda"]

        clash_sub_dict = make_var_to_matlab_sub_dict(variables, clash_prefix="param_")

        self.assertTrue(all([x in clash_sub_dict for x in variables]))
        self.assertTrue(
            all([get_name(x).startswith("param_") for x in clash_sub_dict.values()])
        )

        valid_variables = [sp.Symbol("Y"), TimeAwareSymbol("C", 1), "shocks"]
        clash_sub_dict = make_var_to_matlab_sub_dict(
            valid_variables, clash_prefix="param_"
        )

        self.assertTrue(all([get_name(k) == v for k, v in clash_sub_dict.items()]))

    def test_convert_var_timings_to_matlab(self):
        test_list = ["C_t+1", "C_t", "C_t-1"]
        answers = ["C(1)", "C", "C(-1)"]
        converted = convert_var_timings_to_matlab(test_list)

        for x, ans in zip(converted, answers):
            self.assertEqual(x, ans)

    def test_write_lines_from_list(self):
        from string import ascii_letters

        file = ""
        items = np.random.choice(list(ascii_letters), size=1000, replace=True).tolist()
        file = write_lines_from_list(items, file, line_max=50)

        file_lines = file.split("\n")
        self.assertTrue(all([len(x.strip()) <= 51 for x in file_lines]))

    def test_make_mod_file(self):
        file_path = os.path.join(
            ROOT, "Test GCNs/One_Block_Simple_1_w_Distributions.gcn"
        )
        model = model_from_gcn(file_path, verbose=False)
        model.steady_state(verbose=False)
        model.solve_model(verbose=False)

        mod_file = make_mod_file(model)
        self.assertTrue(isinstance(mod_file, str))


if __name__ == "__main__":
    unittest.main()
