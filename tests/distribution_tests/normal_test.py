import unittest

from functools import partial

import numpy as np

from scipy.stats import truncnorm

from gEconpy.exceptions.exceptions import (
    IgnoredCloseMatchWarning,
    MultipleParameterDefinitionException,
    UnusedParameterWarning,
)
from gEconpy.parser.parse_distributions import (
    LOWER_BOUND_ALIASES,
    NORMAL_LOC_ALIASES,
    NORMAL_SCALE_ALIASES,
    UPPER_BOUND_ALIASES,
    NormalDistributionParser,
)


class TestNormalDistributionParser(unittest.TestCase):
    def setUp(self):
        self.parser = NormalDistributionParser("alpha")

        def tau_to_sigma(name, value):
            return 1 / value if name in ["tau", "precision"] else value

        self.parse_loc_parameter = partial(
            self.parser._parse_parameter,
            canon_name=self.parser.loc_param_name,
            aliases=NORMAL_LOC_ALIASES,
        )

        self.parse_scale_parameter = partial(
            self.parser._parse_parameter,
            canon_name=self.parser.scale_param_name,
            aliases=NORMAL_SCALE_ALIASES,
            additional_transformation=tau_to_sigma,
        )
        self.parse_lower_bound = partial(
            self.parser._parse_parameter, canon_name="a", aliases=LOWER_BOUND_ALIASES
        )
        self.parse_upper_bound = partial(
            self.parser._parse_parameter, canon_name="b", aliases=UPPER_BOUND_ALIASES
        )

    def test_parse_loc_parameter(self):
        parsed_dict = self.parse_loc_parameter({"loc": "0", "scale": "1"})
        self.assertEqual(parsed_dict, {"loc": 0.0})

        parsed_dict = self.parse_loc_parameter({"mu": "0", "std": "1"})
        self.assertEqual(parsed_dict, {"loc": 0.0})

    def test_typo_in_loc_parameter(self):
        param_dict = {"muu": "0", "sd": "1"}

        self.assertWarns(IgnoredCloseMatchWarning, self.parse_loc_parameter, param_dict)

    def test_loc_declared_twice(self):
        param_dict = {"mu": "0", "loc": "1"}

        self.assertRaises(
            MultipleParameterDefinitionException, self.parse_loc_parameter, param_dict
        )

    def test_no_loc_parameter(self):
        d = self.parser.build_distribution({"scale": "3"})
        self.assertEqual(d.mean(), 0)

    def test_parse_scale_parameter(self):
        parsed_dict = self.parse_scale_parameter({"loc": "0", "scale": "1"})
        self.assertEqual(parsed_dict, {"scale": 1.0})

        parsed_dict = self.parse_scale_parameter({"mu": "0", "sigma": "0.25"})
        self.assertEqual(parsed_dict, {"scale": 0.25})

    def test_typo_in_scale_parameter(self):
        param_dict = {"mean": "0", "siggma": "2.5"}
        self.assertWarns(
            IgnoredCloseMatchWarning, self.parse_scale_parameter, param_dict
        )

    def test_scale_declared_twice(self):
        param_dict = {"sigma": "0", "tau": "1"}

        self.assertRaises(
            MultipleParameterDefinitionException, self.parse_scale_parameter, param_dict
        )

    def test_no_scale_parameter(self):
        d = self.parser.build_distribution({"mu": "0"})
        self.assertEqual(d.std(), 1.0)

    def test_parse_bounds(self):
        lower_bound = self.parse_lower_bound({"min": "3"})
        upper_bound = self.parse_upper_bound({"lower_bound": "0", "upper_bound": "10"})

        self.assertEqual(lower_bound, {"a": 3.0})
        self.assertEqual(upper_bound, {"b": 10.0})

    def test_typo_in_bounds(self):
        param_dict = {"loc": 1, "scale": 2, "maxx": 3}
        self.assertWarns(IgnoredCloseMatchWarning, self.parse_upper_bound, param_dict)

        param_dict = {"loc": 1, "scale": 2, "lower___bound": 3}
        self.assertWarns(IgnoredCloseMatchWarning, self.parse_lower_bound, param_dict)

    def test_bounds_declared_twice(self):
        param_dict = {"loc": 1, "max": 2, "upper_bound": 3}
        self.assertRaises(
            MultipleParameterDefinitionException, self.parse_upper_bound, param_dict
        )

        param_dict = {"loc": 1, "lower": 2, "lower_bound": 3}

        self.assertRaises(
            MultipleParameterDefinitionException, self.parse_lower_bound, param_dict
        )

    def test_unused_parameter_warning(self):
        parser = self.parser
        param_dict = {
            "loc": "0",
            "scale": "1",
            "lower_bound": "0",
            "upper_bound": "10",
            "x": "3",
            "y": "4",
            "z": "6",
        }

        self.assertWarns(UnusedParameterWarning, parser.build_distribution, param_dict)

    def test_unbounded_loc_scale(self):
        parser = self.parser
        d = parser.build_distribution({"loc": "0", "scale": "1"})
        self.assertEqual(d.mean(), 0)
        self.assertEqual(d.std(), 1)

    def test_unbounded_mean_std(self):
        parser = self.parser
        d = parser.build_distribution({"mean": "1", "std": "3"})
        self.assertEqual(d.mean(), 1)
        self.assertEqual(d.std(), 3)

    def test_unbounded_loc_std(self):
        parser = self.parser
        d = parser.build_distribution({"loc": "1", "std": "3"})
        self.assertEqual(d.mean(), 1)
        self.assertEqual(d.std(), 3)

    def test_unbounded_mean_scale(self):
        parser = self.parser
        d = parser.build_distribution({"mean": "1", "scale": "3"})
        self.assertEqual(d.mean(), 1)
        self.assertEqual(d.std(), 3)

    def test_bounded_loc_scale(self):
        parser = self.parser
        d = parser.build_distribution({"loc": "0", "scale": "1", "min": "0"})
        d_ = truncnorm(loc=0, scale=1, a=0, b=np.inf)

        self.assertEqual(d.mean(), d_.mean())
        self.assertEqual(d.std(), d_.std())

    def test_bounded_mean_std(self):
        parser = self.parser
        d = parser.build_distribution({"mean": "1", "std": "1", "min": "0"})

        self.assertAlmostEqual(d.mean(), 1, places=2)
        self.assertAlmostEqual(d.std(), 1, places=2)

    def test_bounded_loc_std(self):
        parser = self.parser
        d = parser.build_distribution({"loc": "1", "std": "3", "min": "0"})
        self.assertAlmostEqual(d.std(), 3)

    def test_bounded_mean_scale(self):
        parser = self.parser
        d = parser.build_distribution({"mean": "1", "scale": "3", "min": "0"})
        self.assertAlmostEqual(d.mean(), 1)


if __name__ == "__main__":
    unittest.main()
