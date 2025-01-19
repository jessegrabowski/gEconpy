from typing import Literal

from preliz.distributions.distributions import Distribution
from pyparsing import ParseException

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.exceptions import (
    InvalidDistributionException,
    InvalidParameterException,
    RepeatedParameterException,
)
from gEconpy.parser.dist_syntax import dist_syntax, evaluate_expression
from gEconpy.utilities import is_number

DIST_TO_PARAM_NAMES = {
    "AsymmetricLaplace": ["kappa", "mu", "b", "q"],
    "Bernoulli": ["p", "logit_p"],
    "Beta": ["alpha", "beta", "mu", "sigma", "nu"],
    "BetaBinomial": ["alpha", "beta", "n"],
    "BetaScaled": ["alpha", "beta", "lower", "upper"],
    "Binomial": ["n", "p"],
    "Categorical": ["p", "logit_p"],
    "Cauchy": ["alpha", "beta"],
    "ChiSquared": ["nu"],
    "Dirichlet": ["alpha"],
    "DiscreteUniform": ["lower", "upper"],
    "DiscreteWeibull": ["q", "beta"],
    "ExGaussian": ["mu", "sigma", "nu"],
    "Exponential": ["lam", "beta"],
    "Gamma": ["alpha", "beta", "mu", "sigma"],
    "Geometric": ["p"],
    "Gumbel": ["mu", "beta"],
    "HalfCauchy": ["beta"],
    "HalfNormal": ["sigma", "tau"],
    "HalfStudentT": ["nu", "sigma", "lam"],
    "HyperGeometric": ["N", "k", "n"],
    "InverseGamma": ["alpha", "beta", "mu", "sigma"],
    "Kumaraswamy": ["a", "b"],
    "Laplace": ["mu", "b"],
    "LogLogistic": ["alpha", "beta"],
    "LogNormal": ["mu", "sigma"],
    "Logistic": ["mu", "s"],
    "LogitNormal": ["mu", "sigma", "tau"],
    "Moyal": ["mu", "sigma"],
    "MvNormal": ["mu", "cov", "tau"],
    "NegativeBinomial": ["mu", "alpha", "p", "n"],
    "Normal": ["mu", "sigma", "tau"],
    "Pareto": ["alpha", "m"],
    "Poisson": ["mu"],
    "Rice": ["nu", "sigma", "b"],
    "SkewNormal": ["mu", "sigma", "alpha", "tau"],
    "SkewStudentT": ["mu", "sigma", "a", "b", "lam"],
    "StudentT": ["nu", "mu", "sigma", "lam"],
    "Triangular": ["lower", "c", "upper"],
    "TruncatedNormal": ["mu", "sigma", "lower", "upper"],
    "Uniform": ["lower", "upper"],
    "VonMises": ["mu", "kappa"],
    "Wald": ["mu", "lam", "phi"],
    "Weibull": ["alpha", "beta"],
    "ZeroInflatedBinomial": ["psi", "n", "p"],
    "ZeroInflatedNegativeBinomial": ["psi", "mu", "alpha", "p", "n"],
    "ZeroInflatedPoisson": ["psi", "mu"],
}

WRAPPER_TO_ARGS = {
    "maxent": ["lower", "upper", "mass"],
    "Censored": ["lower", "upper"],
    "Hurdle": ["psi"],
}


def squeeze_list(lst):
    if len(lst) != 1:
        return lst
    return squeeze_list(lst[0])


def _process_kwarg_results(result_dict, variable_name, dist_name, valid_params):
    res = {}
    result_dict = squeeze_list(result_dict)

    for param in result_dict:
        param_name, param_value = param[0], param[1]
        if param_name in res.keys():
            raise RepeatedParameterException(variable_name, dist_name, param_name)
        if param_name not in valid_params:
            raise InvalidParameterException(
                variable_name, dist_name, param_name, valid_params
            )

        res[param_name] = evaluate_expression(param_value)

    return res


def preprocess_distribution_string(
    variable_name: str, d_string: str
) -> tuple[
    tuple[str, dict[str, int | float | None]],
    tuple[str | None, dict[str, int | float | None] | None],
]:
    """
    Convert raw output from pyparsing into a structured format for further processing.

    Parameters
    ----------
    variable_name: str
        A string representing the model parameter associated with this probability distribution.
    d_string: str
        A string representing a probability distribution, extracted from a GCN file by the gEcon_parser.preprocess_gcn
        function.

    Returns
    -------
    dist_info: tuple of str, dict
        A tuple containing the name of the distribution and a dictionary of parameter: value pairs.
    wrapper_info: tuple of str, dict
        A tuple containing the name of the wrapper function and a dictionary of parameter: value pairs.
    """
    try:
        print(d_string)
        parsed_result = dist_syntax.parseString(d_string)
    except ParseException as e:
        raise InvalidDistributionException(variable_name, d_string) from e

    dist_name = parsed_result["dist_name"]
    dist_kwargs = parsed_result["dist_kwargs"]

    if dist_name not in DIST_TO_PARAM_NAMES.keys():
        raise InvalidDistributionException(variable_name, d_string)

    dist_param_dict = _process_kwarg_results(
        dist_kwargs, variable_name, dist_name, DIST_TO_PARAM_NAMES[dist_name]
    )
    dist_param_dict["initial_value"] = evaluate_expression(
        parsed_result["initial_value"]
    )

    # If the variable ends with [] it's a shock, which cannot be given an initial value
    if variable_name.endswith("[]") and dist_param_dict["initial_value"] is not None:
        raise InvalidDistributionException(variable_name, d_string)

    dist_info = (dist_name, dist_param_dict)

    wrapper_name = parsed_result["wrapper_func"]
    wrapper_kwargs = parsed_result["wrapper_kwargs"]

    if wrapper_name is None:
        wrapper_kwargs = {}
        wrapper_info = (wrapper_name, wrapper_kwargs)
        return dist_info, wrapper_info

    if wrapper_name not in WRAPPER_TO_ARGS.keys():
        raise ValueError(
            f'Unknown distribution wrapper {wrapper_name}. Valid functions are '
            f'{", ".join(WRAPPER_TO_ARGS.keys())} '
        )

    wrapper_param_kwargs = _process_kwarg_results(
        wrapper_kwargs, variable_name, dist_name, WRAPPER_TO_ARGS[wrapper_name]
    )

    return (dist_name, dist_param_dict), (wrapper_name, wrapper_param_kwargs)


def preprocess_prior_dict(
    raw_prior_dict: dict[str, str],
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """

    Parameters
    ----------
    raw_prior_dict: dict
        Dictionary of variable name: raw distribution string pairs.

    Returns
    -------
    list of (variable_name, distribution_name, prior_dict) tuples.
        The prior_dict of each variable has parameter names as the keys and parameter values as the values.
        Values are still represented as strings, since we still need to check for compound distributions at a later
        stage.
    """

    variable_names = []
    d_names = []
    param_dicts = []
    for variable_name, d_string in raw_prior_dict.items():
        d_name, param_dict = preprocess_distribution_string(variable_name, d_string)
        variable_names.append(variable_name)
        d_names.append(d_name)
        param_dicts.append(param_dict)

    return variable_names, d_names, param_dicts


def parsed_string_to_preliz(
    variable_name: str,
    d_name: str,
    param_dict: dict[str, str],
) -> Distribution:
    """
    Parameters
    ----------
    variable_name: str
        name of the variable with which this distribution is associated
    d_name: str
        plaintext name of the distribution to parameterize, from the CANNON_NAMES list.
    param_dict: dict
        a dictionary of parameter: value pairs, or parameter: string pairs in the case of composite distributions
    backend: str
        backend of the distribution function to parameterize

    Returns
    -------
    d: Distribution
        Preliz Distribution object
    """

    param_dict.pop("initial_value", None)
    param_dict.pop("wrapper_func", None)
    param_dict.pop("wrapper_args", {})


def param_values_to_floats(param_dict: dict):
    for param, param_value in param_dict.items():
        if isinstance(param_value, str):
            if is_number(param_value):
                param_dict[param] = float(param_value)

    return param_dict


def fetch_rv_params(param_dict, model):
    return_dict = {}
    for k, v in param_dict.items():
        if isinstance(v, float | int):
            return_dict[k] = v
        elif isinstance(v, str):
            return_dict[k] = model[v]
        else:
            raise ValueError(
                f"Found an illegal key:value pair in prior param dict, {k}:{v}"
            )

    return return_dict


def create_prior_distribution_dictionary(
    raw_prior_dict: dict[str, str], backend: Literal["scipy", "pymc"] = "scipy"
) -> tuple[SymbolDictionary, SymbolDictionary]:
    """

    Parameters
    ----------
    raw_prior_dict: dict[str, str]
        Dictionary of variable name: distribution string pairs.

    backend: Literal['scipy', 'pymc']
        Which backend to use to create the distributions. Currently "scipy" and "pymc" are supported.

    Returns
    -------
    prior_dict: SymbolDictionary
        A dictionary of variable name: distribution pairs.

    hyper_prior_dict: SymbolDictionary
        A dictionary of variable name: (parent_rv, param, rv) pairs. This is used to keep track of the parent-child
        relationships between distributions in the case of compound distributions.
    """
    variable_names, d_names, param_dicts = preprocess_prior_dict(raw_prior_dict)
    # basic_distributions, compound_distributions = split_out_composite_distributions(
    #     variable_names, d_names, param_dicts
    # )
    #
    # SymbolDictionary()
    # SymbolDictionary()
    #
    # for variable_name, (d_name, param_dict) in basic_distributions.items():
    #     d = distribution_factory(
    #         variable_name=variable_name, d_name=d_name, param_dict=param_dict
    #     )
    #     prior_dict[variable_name] = d
    #
    # for variable_name, (d_name, param_dict) in compound_distributions.items():
    #     rvs_used_in_d = []
    #     for param, value in param_dict.items():
    #         if value in prior_dict.keys():
    #             param_dict[param] = prior_dict[value]
    #             rvs_used_in_d.append((variable_name, param, value))
    #
    #     d = composite_distribution_factory(
    #         variable_name, d_name, param_dict, backend=backend
    #     )
    #     prior_dict[variable_name] = d
    #     for parent_rv, param, rv in rvs_used_in_d:
    #         hyper_prior_dict[rv] = (parent_rv, param, prior_dict[rv])
    #         del prior_dict[rv]
    #
    # return prior_dict, hyper_prior_dict
