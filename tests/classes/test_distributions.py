import preliz as pz

from gEconpy.classes.distributions import CompositeDistribution, priors_by_sampled_name


def test_priors_by_sampled_name_keys_a_shock_hyper_parameter_by_its_hyper_name():
    alpha = pz.Beta(alpha=2, beta=5)
    sigma = pz.Gamma(alpha=2, beta=10)
    estimated = CompositeDistribution(
        name="epsilon_A",
        dist_name="Normal",
        fixed_params={"mu": 0.0},
        hyper_param_dict={"sigma": sigma},
        param_name_to_hyper_name={"sigma": "sigma_epsilon_A"},
    )
    fixed = CompositeDistribution(
        name="epsilon_B",
        dist_name="Normal",
        fixed_params={"mu": 0.0, "sigma": 0.01},
        hyper_param_dict={},
        param_name_to_hyper_name={},
    )

    priors = priors_by_sampled_name(
        param_priors={"alpha": alpha}, shock_priors={"epsilon_A": estimated, "epsilon_B": fixed}
    )

    assert list(priors.items()) == [("alpha", alpha), ("sigma_epsilon_A", sigma)]
