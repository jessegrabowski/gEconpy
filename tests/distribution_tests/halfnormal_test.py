import unittest

from functools import partial

from scipy.stats import halfnorm

from gEconpy.exceptions import (
    IgnoredCloseMatchWarning,
    MultipleParameterDefinitionException,
    UnusedParameterWarning,
)
from gEconpy.parser.parse_distributions import (
    NORMAL_SCALE_ALIASES,
    HalfNormalDistributionParser,
)


class TestHalfNormalDistributionParser(unittest.TestCase):
    def setUp(self):
        self.parser = HalfNormalDistributionParser("alpha")

        def tau_to_sigma(name, value):
            return 1 / value if name in ["tau", "precision"] else value

        self.parse_loc_parameter = partial(
            self.parser._parse_parameter,
            canon_name=self.parser.loc_param_name,
            aliases=[self.parser.loc_param_name],
        )

        self.parse_scale_parameter = partial(
            self.parser._parse_parameter,
            canon_name=self.parser.scale_param_name,
            aliases=NORMAL_SCALE_ALIASES,
            additional_transformation=tau_to_sigma,
        )

    def test_parse_loc_parameter(self):
        parsed_dict = self.parse_loc_parameter({"loc": "0", "scale": "1"})
        self.assertEqual(parsed_dict, {"loc": 0.0})

    def test_typo_in_loc_parameter(self):
        param_dict = {"locc": "0", "sd": "1"}

        self.assertWarns(IgnoredCloseMatchWarning, self.parse_loc_parameter, param_dict)

    def test_parse_scale_parameter(self):
        parsed_dict = self.parse_scale_parameter({"loc": "0", "scale": "1"})
        self.assertEqual(parsed_dict, {"scale": 1.0})

        parsed_dict = self.parse_scale_parameter({"loc": "0", "sigma": "0.25"})
        self.assertEqual(parsed_dict, {"scale": 0.25})

        parsed_dict = self.parse_scale_parameter({"loc": "0", "tau": "0.25"})
        self.assertEqual(parsed_dict, {"scale": 4})

    def test_typo_in_scale_parameter(self):
        param_dict = {"siggma": "2.5"}
        self.assertWarns(
            IgnoredCloseMatchWarning, self.parse_scale_parameter, param_dict
        )

    def test_scale_declared_twice(self):
        param_dict = {"sigma": "0", "tau": "1"}

        self.assertRaises(
            MultipleParameterDefinitionException, self.parse_scale_parameter, param_dict
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

    def test_loc_scale(self):
        parser = self.parser
        d = parser.build_distribution({"loc": "0", "scale": "1"})
        d_ = halfnorm(loc=0, scale=1)

        self.assertEqual(d.mean(), d_.mean())
        self.assertEqual(d.std(), d_.std())

    def test_mean_std(self):
        parser = self.parser
        d = parser.build_distribution({"mean": "1", "std": "3"})
        self.assertAlmostEqual(d.mean(), 1)
        self.assertAlmostEqual(d.std(), 3)

    def test_loc_std(self):
        parser = self.parser
        d = parser.build_distribution({"loc": "1", "std": "3"})
        self.assertAlmostEqual(d.std(), 3)

    def test_unbounded_mean_scale(self):
        parser = self.parser
        d = parser.build_distribution({"mean": "1", "scale": "3"})
        self.assertAlmostEqual(d.mean(), 1)


if __name__ == "__main__":
    unittest.main()
