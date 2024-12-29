import unittest

from functools import partial

from gEconpy.exceptions import (
    IgnoredCloseMatchWarning,
    InvalidParameterException,
    UnusedParameterWarning,
)
from gEconpy.parser.parse_distributions import (
    EXPONENTIAL_RATE_ALIASES,
    ExponentialDistributionParser,
)


class TestExponentialDistributionParser(unittest.TestCase):
    def setUp(self):
        self.parser = ExponentialDistributionParser("alpha")

        self.parse_loc_parameter = partial(
            self.parser._parse_parameter,
            canon_name=self.parser.loc_param_name,
            aliases=[self.parser.loc_param_name],
        )

        self.parse_scale_parameter = partial(
            self.parser._parse_parameter,
            canon_name=self.parser.scale_param_name,
            aliases=[self.parser.scale_param_name],
        )
        self.parse_rate_parameter = partial(
            self.parser._parse_parameter,
            canon_name=self.parser.rate_param_name,
            aliases=EXPONENTIAL_RATE_ALIASES,
        )

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

        parsed_dict = self.parse_scale_parameter({"scale": "0.5"})
        self.assertEqual(parsed_dict, {"scale": 0.5})

    def test_typo_in_scale_parameter(self):
        param_dict = {"mean": "0", "scaale": "2.5"}
        self.assertWarns(
            IgnoredCloseMatchWarning, self.parse_scale_parameter, param_dict
        )

    def test_unused_parameter_warning(self):
        parser = self.parser
        param_dict = {"rate": "1", "x": "3", "y": "4", "z": "6"}

        self.assertWarns(UnusedParameterWarning, parser.build_distribution, param_dict)

    def test_distribution_from_moments(self):
        parser = self.parser
        d = parser.build_distribution({"sd": "0.1", "mean": "3"})

        self.assertAlmostEqual(d.mean(), 3)
        self.assertAlmostEqual(d.std(), 10)


if __name__ == "__main__":
    unittest.main()
