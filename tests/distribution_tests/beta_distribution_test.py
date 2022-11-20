import unittest
import numpy as np
from scipy.stats import beta

from gEconpy.exceptions.exceptions import MultipleParameterDefinitionException, InvalidParameterException, \
    IgnoredCloseMatchWarning, UnusedParameterWarning
from gEconpy.parser.parse_distributions import BetaDistributionParser, BETA_SHAPE_ALIASES_1, \
    BETA_SHAPE_ALIASES_2

from functools import partial


class TestBetaDistributionParser(unittest.TestCase):

    def setUp(self):
        self.parser = BetaDistributionParser('alpha')

        self.parse_loc_parameter = partial(self.parser._parse_parameter,
                                           canon_name=self.parser.loc_param_name,
                                           aliases=[self.parser.loc_param_name])
        self.parse_scale_parameter = partial(self.parser._parse_parameter,
                                             canon_name=self.parser.scale_param_name,
                                             aliases=[self.parser.scale_param_name])
        self.parse_shape_parameter_1 = partial(self.parser._parse_parameter,
                                               canon_name=self.parser.shape_param_name_1,
                                               aliases=BETA_SHAPE_ALIASES_1)
        self.parse_shape_parameter_2 = partial(self.parser._parse_parameter,
                                               canon_name=self.parser.shape_param_name_2,
                                               aliases=BETA_SHAPE_ALIASES_2)

    def test_parse_loc_parameter(self):
        parsed_dict = self.parse_loc_parameter({'loc': '1', 'sd': '1'})
        self.assertEqual(parsed_dict, {'loc': 1})

        self.parser._parse_mean_constraint({'mean': '3'})
        self.assertEqual(self.parser.mean_constraint, 3)

    def test_typo_in_loc_parameter(self):
        param_dict = {'loocc': '0', 'sd': '1'}

        self.assertWarns(IgnoredCloseMatchWarning,
                         self.parse_loc_parameter,
                         param_dict)

    def test_parse_scale_parameter(self):
        self.parser._parse_std_constraint({'std': '2'})
        self.assertEqual(self.parser.std_constraint, 2)

        parsed_dict = self.parse_scale_parameter({'scale': '1'})
        self.assertEqual(parsed_dict, {'scale': 1})

    def test_typo_in_scale_parameter(self):
        param_dict = {'mean': '0', 'scaale': '2.5'}
        self.assertWarns(IgnoredCloseMatchWarning,
                         self.parse_scale_parameter,
                         param_dict)

    def test_unused_parameter_warning(self):
        parser = self.parser
        param_dict = {'alpha': '1', 'beta': '1', 'x': '3', 'y': '4', 'z': '6'}

        self.assertWarns(UnusedParameterWarning,
                         parser.build_distribution,
                         param_dict)

    def test_parse_alpha(self):
        parsed_dict = self.parse_shape_parameter_1({'loc': '1', 'sd': '1', 'alpha': 1})
        self.assertEqual(parsed_dict, {'a': 1})

    def test_typo_in_alpha(self):
        param_dict = {'alppha': '0', 'sd': '1'}

        self.assertWarns(IgnoredCloseMatchWarning,
                         self.parse_shape_parameter_1,
                         param_dict)

    def test_moment_inequality_error(self):
        parser = self.parser
        self.assertRaises(InvalidParameterException,
                          parser.build_distribution,
                          {'sd': '2', 'mean': '0.5'})

    def test_distribution_from_moments(self):
        parser = self.parser
        d = parser.build_distribution({'sd': '0.1', 'mean': '0.5'})

        self.assertAlmostEqual(d.std(), 0.1)
        self.assertAlmostEqual(d.mean(), 0.5)


if __name__ == '__main__':
    unittest.main()
