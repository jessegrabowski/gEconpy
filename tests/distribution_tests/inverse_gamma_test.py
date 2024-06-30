import unittest
from functools import partial

import numpy as np

from gEconpy.exceptions.exceptions import (
    IgnoredCloseMatchWarning,
    MultipleParameterDefinitionException,
    UnusedParameterWarning,
)
from gEconpy.parser.parse_distributions import (
    INV_GAMMA_SCALE_ALIASES,
    INV_GAMMA_SHAPE_ALIASES,
    InverseGammaDistributionParser,
)


class TestInverseGammaDistributionParser(unittest.TestCase):
    def setUp(self):
        self.parser = InverseGammaDistributionParser("alpha")

        self.parse_loc_parameter = partial(
            self.parser._parse_parameter,
            canon_name=self.parser.loc_param_name,
            aliases=[self.parser.loc_param_name],
        )
        self.parse_scale_parameter = partial(
            self.parser._parse_parameter,
            canon_name=self.parser.scale_param_name,
            aliases=INV_GAMMA_SCALE_ALIASES,
        )
        self.parse_shape_parameter = partial(
            self.parser._parse_parameter,
            canon_name=self.parser.shape_param_name,
            aliases=INV_GAMMA_SHAPE_ALIASES,
        )

    #         self.valid_param_dict_1 =
    #         self.valid_param_dict_2 =
    #         self.valid_param_dict_3 = {'alpha': '1', 'beta': '3', 'loc': '3'}

    def test_parse_loc_parameter(self):
        parsed_dict = self.parse_loc_parameter({"loc": "1", "sd": "1"})
        self.assertEqual(parsed_dict, {"loc": 1})

        self.parser._parse_mean_constraint({"mean": "3"})
        self.assertEqual(self.parser.mean_constraint, 3)

    def test_typo_in_loc_parameter(self):
        param_dict = {"loocc": "0", "sd": "1"}

        self.assertWarns(IgnoredCloseMatchWarning, self.parse_loc_parameter, param_dict)

    def test_parse_scale_parameter(self):
        self.parser._parse_std_constraint({"std": "2"})
        self.assertEqual(self.parser.std_constraint, 2)

        parsed_dict = self.parse_scale_parameter({"alpha": "1", "beta": "2"})
        self.assertEqual(parsed_dict, {"scale": 2.0})

    def test_typo_in_scale_parameter(self):
        param_dict = {"mean": "0", "betta": "2.5"}
        self.assertWarns(
            IgnoredCloseMatchWarning, self.parse_scale_parameter, param_dict
        )

    def test_scale_declared_twice(self):
        param_dict = {"beta": "0", "b": "1"}

        self.assertRaises(
            MultipleParameterDefinitionException, self.parse_scale_parameter, param_dict
        )

    def test_unused_parameter_warning(self):
        parser = self.parser
        param_dict = {"scale": "1", "shape": "1", "x": "3", "y": "4", "z": "6"}

        self.assertWarns(UnusedParameterWarning, parser.build_distribution, param_dict)

    def test_distribution_from_moments(self):
        parser = self.parser
        d = parser.build_distribution({"sd": "3", "mean": "2"})

        self.assertAlmostEqual(d.std(), 3)
        self.assertAlmostEqual(d.mean(), 2)

    def test_distribution_from_loc_scale(self):
        parser = self.parser
        d = parser.build_distribution({"loc": "1", "scale": "1", "a": "3"})

        self.assertEqual(d.mean(), 1.5)
        self.assertEqual(d.std(), 0.5)

    def test_distribution_from_shape_and_std(self):
        parser = self.parser
        d = parser.build_distribution({"std": "0.5", "a": "3"})
        a, std = 3, 0.5
        b = std * (a - 1) * np.sqrt(a - 2)

        self.assertEqual(d.std(), 0.5)
        self.assertEqual(b**2 / (a - 1) ** 2 / (a - 2), d.var())

    def test_distribution_from_scale_and_std(self):
        parser = self.parser
        d = parser.build_distribution({"scale": "2", "std": "3"})
        self.assertAlmostEqual(d.std(), 3)
        self.assertAlmostEqual(d.mean(), 1.57000653240468)

    def test_distribution_from_shape_and_mean(self):
        parser = self.parser
        d = parser.build_distribution({"mean": "1", "a": "3"})
        self.assertAlmostEqual(d.mean(), 1)

    def test_distribution_from_scale_and_mean(self):
        parser = self.parser
        d = parser.build_distribution({"mean": "1", "scale": "1"})
        self.assertAlmostEqual(d.mean(), 1)


if __name__ == "__main__":
    unittest.main()
