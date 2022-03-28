import unittest

from scipy.stats import norm, invgamma

from gEcon.exceptions.exceptions import InvalidDistributionException, DistributionParsingError, \
    MissingParameterValueException, RepeatedParameterException
from gEcon.parser.gEcon_parser import preprocess_gcn
from gEcon.parser.parse_distributions import preprocess_distribution_string, distribution_factory, \
    create_prior_distribution_dictionary


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
        self.assertEqual(list(prior_dict.values()), ['norm(mu = 0, sd = 1)',
                                                     'N(mean = 0, sd = 1)'])

    def test_catch_no_initial_value(self):
        no_initial_value = '''
                Block TEST
                {                
                    calibration
                    {
                        alpha ~ N(mean = 0, sd = 1);
                    };
                };
        '''

        self.assertRaises(MissingParameterValueException,
                          preprocess_gcn,
                          no_initial_value)

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
        dicts = [{'mu': '0', 'sd': '1'}, {'mean': '0', 'sd': '1'}]

        for i, (param_name, distribution_string) in enumerate(prior_dict.items()):
            dist_name, param_dict = preprocess_distribution_string(param_name, distribution_string)

            self.assertEqual(dist_name, 'normal')
            self.assertEqual(param_dict, dicts[i])

    def test_parse_compound_distributions(self):
        compound_distribution = '''Block TEST
                {
                    calibration
                    {   
                        sigma_alpha ~ inv_gamma(a=20, scale=1) = 0.01;
                        mu_alpha ~ N(mean = 1, scale=1) = 0.01;
                        alpha ~ N(mean = mu_alpha, sd = sigma_alpha) = 0.5;
                    };
                };'''

        model, raw_prior_dict = preprocess_gcn(compound_distribution)
        prior_dict = create_prior_distribution_dictionary(raw_prior_dict)

        d = prior_dict['alpha']

        self.assertEqual(d.rv_params['loc'].mean(), 1)
        self.assertEqual(d.rv_params['loc'].std(), 1)
        self.assertEqual(d.rv_params['scale'].mean(), 1 / (20 - 1))
        self.assertEqual(d.rv_params['scale'].var(), 1 ** 2 / (20 - 1) ** 2 / (20 - 2))

class TestDistributionFactory(unittest.TestCase):

    def test_parse_distributions(self):
        file = '''
            TEST_BLOCK
            {
                shocks
                {
                    epsilon[] ~ N(mean=0, std=0.1);
                };
                
                calibration
                {
                    alpha ~ beta(a=1, b=1) = 0.5;
                    rho ~ gamma(mean=0.95, sd=1) = 0.95;
                    sigma ~ inv_gamma(mean=0.01, sd=0.1) = 0.01;
                    tau ~ halfnorm(mean=0.5, sd=1) = 1;
                    psi ~ norm(mean=1.5, sd=1.5, min=0) = 1;
                };
            };
        '''

        model, prior_dict = preprocess_gcn(file)
        means = [0, 0.5, 0.95, 0.01, 0.5, 1.5]
        stds = [0.1, 0.28867513459481287, 1, 0.1, 1, 1.5]

        for i, (variable_name, d_string) in enumerate(prior_dict.items()):
            d_name, param_dict = preprocess_distribution_string(variable_name, d_string)
            d = distribution_factory(variable_name=variable_name, d_name=d_name, param_dict=param_dict)
            self.assertAlmostEqual(d.mean(), means[i], places=3)
            self.assertAlmostEqual(d.std(), stds[i], places=3)





if __name__ == '__main__':
    unittest.main()
