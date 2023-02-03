import unittest
from functools import partial

import numpy as np
from scipy.stats import gamma

from gEconpy.exceptions.exceptions import (
    IgnoredCloseMatchWarning,
    MultipleParameterDefinitionException,
    UnusedParameterWarning,
)
from gEconpy.parser.parse_distributions import (
    GAMMA_SCALE_ALIASES,
    GAMMA_SHAPE_ALIASES,
    GammaDistributionParser,
)


class TestBetaDistributionParser(unittest.TestCase):
    def setUp(self):
        self.parser = GammaDistributionParser("alpha")

        self.parse_loc_parameter = partial(
            self.parser._parse_parameter,
            canon_name=self.parser.loc_param_name,
            aliases=[self.parser.loc_param_name],
        )
        self.parse_scale_parameter = partial(
            self.parser._parse_parameter,
            canon_name=self.parser.scale_param_name,
            aliases=GAMMA_SCALE_ALIASES,
        )
        self.parse_shape_parameter = partial(
            self.parser._parse_parameter,
            canon_name=self.parser.shape_param_name,
            aliases=GAMMA_SHAPE_ALIASES,
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

        parsed_dict = self.parse_scale_parameter({"beta": "1"})
        self.assertEqual(parsed_dict, {"scale": 1})

    def test_typo_in_scale_parameter(self):
        param_dict = {"mean": "0", "sccale": "2.5"}
        self.assertWarns(IgnoredCloseMatchWarning, self.parse_scale_parameter, param_dict)

    def test_unused_parameter_warning(self):
        parser = self.parser
        param_dict = {"shape": "1", "beta": "1", "x": "3", "y": "4", "z": "6"}

        self.assertWarns(UnusedParameterWarning, parser.build_distribution, param_dict)

    def test_parse_alpha(self):
        parsed_dict = self.parse_shape_parameter({"loc": "1", "sd": "1", "alpha": 1})
        self.assertEqual(parsed_dict, {"a": 1})

    def test_typo_in_alpha(self):
        param_dict = {"alppha": "0", "sd": "1"}

        self.assertWarns(IgnoredCloseMatchWarning, self.parse_shape_parameter, param_dict)

    def test_distribution_from_moments(self):
        parser = self.parser
        d = parser.build_distribution({"sd": "3", "mean": "0.5"})

        self.assertAlmostEqual(d.std(), 3)
        self.assertAlmostEqual(d.mean(), 0.5)

    def test_distribution_from_scale_and_shape(self):
        parser = self.parser
        d = parser.build_distribution({"alpha": "0.5", "beta": "2"})

        self.assertAlmostEqual(d.mean(), 0.5 * 2)
        self.assertAlmostEqual(d.std(), np.sqrt(0.5 * 2**2))

    def test_distribution_from_scale_and_mean(self):
        parser = self.parser
        d = parser.build_distribution({"scale": "1", "mean": "3"})

        self.assertAlmostEqual(d.mean(), 3)

    def test_distribution_from_shape_and_std(self):
        parser = self.parser
        d = parser.build_distribution({"alpha": "1", "std": "3"})

        self.assertAlmostEqual(d.std(), 3)

    def test_distribution_from_shape_and_mean(self):
        parser = self.parser
        d = parser.build_distribution({"alpha": "1", "mean": "3"})

        self.assertAlmostEqual(d.mean(), 3)


if __name__ == "__main__":
    unittest.main()
