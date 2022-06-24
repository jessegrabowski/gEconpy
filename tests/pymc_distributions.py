import unittest

from gEcon.exceptions.exceptions import InvalidDistributionException, DistributionParsingError, \
    MissingParameterValueException, RepeatedParameterException
from gEcon.parser.gEcon_parser import preprocess_gcn
from gEcon.parser.parse_distributions import preprocess_distribution_string, distribution_factory, build_pymc_model
import pymc as pm


class TestParsePriorsToPyMC(unittest.TestCase):

    def setUp(self):
        self.file = '''
                Block TEST
                {
                    shocks
                    {
                        epsilon[] ~ norm(mu = 0, sd = 1);
                        epsilon_two[] ~ N(mu = 0, sd = sigma_2);
                    };
                    
                    calibration
                    {
                        alpha ~ N(mean = 0, sd = 1) = 0.5;
                        sigma_2 ~ HalfNormal(loc=1.0, sigma=1.0) = 0.01;
                        beta_test ~ Beta(a=1.0, b=1.0) = 0.5;
                        gamma_test ~ Gamma(a=0.5, b=1.0) = 2.0;
                        inv_gamma_test ~ Invgamma(a=3.0, b=1.0) = 2.0;
                        unif_test ~ Uniform(lower=0.0, upper=1.0) = 0.5;
                    };
                };
        '''

    def test_parse_distributions(self):
        model, raw_prior_dict = preprocess_gcn(self.file)
        pymc_model = build_pymc_model(raw_prior_dict, None)

        rvs = pymc_model.rvs_to_values

        self.assertEqual(len(rvs), 8)
        self.assertTrue(all([rv.name in ['epsilon[]', 'epsilon_two[]', 'alpha', 'sigma_2',
                                         'raw_sigma_2', 'beta_test', 'gamma_test', 'inv_gamma_test',
                                         'unif_test'] for rv in rvs]))

        self.assertTrue('sigma_2' in str(pymc_model['epsilon_two[]'].get_parents()[0]))

if __name__ == '__main__':
    unittest.main()
