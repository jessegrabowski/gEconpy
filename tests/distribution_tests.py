import unittest

import numpy as np
from scipy.stats._distn_infrastructure import rv_frozen

from gEcon.exceptions.exceptions import InvalidDistributionException, DistributionParsingError, \
    ParameterNotFoundException, MultipleParameterDefinitionException, RepeatedParameterException, \
    IgnoredCloseMatchWarning, UnusedParameterWarning
from gEcon.parser.gEcon_parser import preprocess_gcn
from gEcon.parser.parse_distributions import preprocess_distribution_string, NormalDistributionParser


class BasicParerFunctionalityTests(unittest.TestCase):

    def setUp(self):
        self.file = '''
                Block TEST
                {
                    shocks
                    {
                        epsilon[] ~ norm(mu = 0, sd = 1);
                    };
                    
                    calibration
                    {
                        alpha ~ N(mean = 0, sd = 1) = 0.5;
                    };
                };
        '''

    def test_extract_param_dist_simple(self):
        model, prior_dict = preprocess_gcn(self.file)
        self.assertEqual(list(prior_dict.keys()), ['epsilon[]', 'alpha'])
        self.assertEqual(list(prior_dict.values()), ['norm(mu = 0, sd = 1)', 'N(mean = 0, sd = 1)'])

    def test_catch_typo_in_param_dist_definition(self):
        squiggle_is_equal = '''
                Block TEST
                {                
                    calibration
                    {
                        alpha = N((mean = 0, sd = 1) = 0.5;
                    };
                };
        '''

        self.assertRaises(DistributionParsingError,
                          preprocess_gcn,
                          squiggle_is_equal)

    def test_catch_distribution_typos(self):
        extra_parenthesis_start = '''
                Block TEST
                {
                    calibration
                    {
                        alpha ~ N((mean = 0, sd = 1) = 0.5;
                    };
                };
        '''

        extra_parenthesis_end = '''
                Block TEST
                {
                    calibration
                    {
                        alpha ~ N(mean = 0, sd = 1)) = 0.5;
                    };
                };
        '''

        extra_equals = '''
                Block TEST
                {
                    calibration
                    {
                        alpha ~ N(mean == 0, sd = 1) = 0.5;
                    };
                };
        '''

        missing_common = '''
                Block TEST
                {
                    calibration
                    {
                        alpha ~ N(mean = 0 sd = 1) = 0.5;
                    };
                };
        '''

        shock_with_starting_value = '''
                Block TEST
                {                
                    shocks
                    {
                        epsilon[] ~ N(mean = 0, sd = 1) = 0.5;
                    };
                };
        '''

        test_files = [extra_parenthesis_start, extra_parenthesis_end, extra_equals, missing_common,
                      shock_with_starting_value]

        for file in test_files:
            model, prior_dict = preprocess_gcn(file)
            for param_name, distribution_string in prior_dict.items():
                self.assertRaises(InvalidDistributionException,
                                  preprocess_distribution_string,
                                  variable_name=param_name,
                                  d_string=distribution_string)

    def test_catch_repeated_parameter_definition(self):
        repeated_parameter = '''
                Block TEST
                {
                    calibration
                    {
                        alpha ~ N(mean = 0, mean = 1) = 0.5;
                    };
                };
        '''
        model, prior_dict = preprocess_gcn(repeated_parameter)

        for param_name, distribution_string in prior_dict.items():
            self.assertRaises(RepeatedParameterException,
                              preprocess_distribution_string,
                              variable_name=param_name,
                              d_string=distribution_string)

    def test_parameter_parsing_simple(self):
        model, prior_dict = preprocess_gcn(self.file)
        names = ['norm', 'N']
        dicts = [{'mu': '0', 'sd': '1'}, {'mean': '0', 'sd': '1'}]

        for i, (param_name, distribution_string) in enumerate(prior_dict.items()):
            dist_name, param_dict = preprocess_distribution_string(param_name, distribution_string)

            self.assertEqual(dist_name, names[i])
            self.assertEqual(param_dict, dicts[i])


class TestNormalDistributionParser(unittest.TestCase):

    def setUp(self):
        self.file = '''
                Block TEST
                {
                    shocks
                    {
                        epsilon[] ~ norm(mu = 0, sd = 1);
                    };
                    
                    calibration
                    {
                        alpha ~ N(mean = 0, sd = 1) = 0.5;
                    };
                };
        '''

        self.valid_param_dict_1 = {'mu': '0', 'sd': '1'}
        self.valid_param_dict_2 = {'mean': '0', 'scale': '0.25', 'min': '0', 'max': '2'}
        self.valid_param_dict_3 = {'loc': '0', 'tau': '2'}

        self.parser = NormalDistributionParser('alpha')

    def test_parse_loc_parameter(self):
        parser = self.parser
        parsed_dict = parser._parse_loc_parameter(self.valid_param_dict_1)
        self.assertEqual(parsed_dict, {'loc': 0.0})

        parsed_dict = parser._parse_loc_parameter(self.valid_param_dict_2)
        self.assertEqual(parsed_dict, {'loc': 0.0})

        parsed_dict = parser._parse_loc_parameter(self.valid_param_dict_3)
        self.assertEqual(parsed_dict, {'loc': 0.0})

    def test_typo_in_loc_parameter(self):
        parser = self.parser
        param_dict = {'muu': '0', 'sd': '1'}

        self.assertRaises(ParameterNotFoundException,
                          parser._parse_loc_parameter,
                          param_dict)

    def test_loc_declared_twice(self):
        parser = self.parser
        param_dict = {'loc': '0', 'mean': '1'}

        self.assertRaises(MultipleParameterDefinitionException,
                          parser._parse_loc_parameter,
                          param_dict)

    def test_parse_scale_parameter(self):
        parser = self.parser
        parsed_dict = parser._parse_scale_parameter(self.valid_param_dict_1)
        self.assertEqual(parsed_dict, {'scale': 1.0})

        parsed_dict = parser._parse_scale_parameter(self.valid_param_dict_2)
        self.assertEqual(parsed_dict, {'scale': 0.25})

        parsed_dict = parser._parse_scale_parameter(self.valid_param_dict_3)
        self.assertEqual(parsed_dict, {'scale': 0.5})

    def test_typo_in_scale_parameter(self):
        parser = self.parser
        param_dict = {'mean': '0', 'stdd': '2.5'}
        self.assertRaises(ParameterNotFoundException,
                          parser._parse_scale_parameter,
                          param_dict)

    def test_scale_declared_twice(self):
        parser = self.parser
        param_dict = {'sd': '0', 'tau': '1'}

        self.assertRaises(MultipleParameterDefinitionException,
                          parser._parse_scale_parameter,
                          param_dict)

    def test_parse_bounds(self):
        parser = self.parser
        lower_bound = parser._parse_lower_bound(self.valid_param_dict_1)
        upper_bound = parser._parse_upper_bound(self.valid_param_dict_1)

        self.assertEqual(lower_bound, {'a': -np.inf})
        self.assertEqual(upper_bound, {'b': np.inf})

        lower_bound = parser._parse_lower_bound(self.valid_param_dict_2)
        upper_bound = parser._parse_upper_bound(self.valid_param_dict_2)

        self.assertEqual(lower_bound, {'a': 0.0})
        self.assertEqual(upper_bound, {'b': 2.0})

    def test_typo_in_bounds(self):
        parser = self.parser
        param_dict = {'loc': 1, 'scale': 2, 'maxx': 3}
        self.assertWarns(IgnoredCloseMatchWarning,
                         parser._parse_upper_bound,
                         param_dict)

        param_dict = {'loc': 1, 'scale': 2, 'lower___bound': 3}

        self.assertWarns(IgnoredCloseMatchWarning,
                         parser._parse_lower_bound,
                         param_dict)

    def test_bounds_declared_twice(self):
        parser = self.parser
        param_dict = {'loc': 1, 'max': 2, 'upper_bound': 3}
        self.assertRaises(MultipleParameterDefinitionException,
                          parser._parse_upper_bound,
                          param_dict)

        param_dict = {'loc': 1, 'lower': 2, 'lower_bound': 3}

        self.assertRaises(MultipleParameterDefinitionException,
                          parser._parse_lower_bound,
                          param_dict)

    def test_unused_parameter_warning(self):
        parser = self.parser
        param_dict = {'loc': '0', 'scale': '1', 'lower_bound': '0', 'upper_bound': '10', 'x': '3', 'y': '4', 'z': '6'}

        self.assertWarns(UnusedParameterWarning,
                         parser.build_distribution,
                         param_dict)

    def test_unbounded_distribution(self):
        parser = self.parser
        d = parser.build_distribution(self.valid_param_dict_1)
        self.assertIsInstance(d, rv_frozen)

        self.assertEqual(d.mean(), float(self.valid_param_dict_1['mu']))
        self.assertEqual(d.std(), float(self.valid_param_dict_1['sd']))

    def test_bounded_distribution(self):
        parser = self.parser
        d = parser.build_distribution(self.valid_param_dict_2)
        self.assertIsInstance(d, rv_frozen)

        self.assertEqual(d.mean(), float(self.valid_param_dict_2['mean']))
        self.assertEqual(d.std(), float(self.valid_param_dict_2['scale']))


if __name__ == '__main__':
    unittest.main()
