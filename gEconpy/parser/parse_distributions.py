import preliz as pz

from preliz.distributions.distributions import Distribution
from pyparsing import ParseException

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.exceptions import (
    InvalidDistributionException,
    RepeatedParameterException,
)
from gEconpy.parser.dist_syntax import (
    PRELIZ_DIST_WRAPPERS,
    PRELIZ_DISTS,
    dist_syntax,
    evaluate_expression,
)
from gEconpy.utilities import is_number


def squeeze_list(lst):
    if len(lst) != 1:
        return lst
    return squeeze_list(lst[0])


def _process_kwarg_results(result_dict, variable_name, dist_name, valid_params):
    res = {}
    # result_dict = squeeze_list(result_dict)

    for param_name, param_value in result_dict.items():
        if param_name in res.keys():
            raise RepeatedParameterException(variable_name, dist_name, param_name)

        res[param_name] = evaluate_expression(param_value)

    return res


def split_prior_dict_by_params_and_shocks(
    raw_prior_dict: dict[str, str],
) -> tuple[dict[str, str], dict[str, tuple[str, dict[str, tuple[str, str]]]]]:
    raw_param_dict = {}
    raw_shock_dict = {}
    shock_hyper_params = []

    for variable_name, d_string in raw_prior_dict.items():
        if variable_name.endswith("[]"):
            dist_hyper_dists = {}
            [parsed_dist] = dist_syntax.parse_string(d_string, parse_all=True)
            dist_kwargs = parsed_dist["dist_kwargs"]

            for param_name, param_value in dist_kwargs.items():
                if isinstance(param_value, str) and param_value in raw_prior_dict:
                    dist_hyper_dists[param_name] = (
                        param_value,
                        raw_prior_dict[param_value],
                    )
                    shock_hyper_params.append(param_value)
                else:
                    dist_hyper_dists[param_name] = (None, param_value)

            raw_shock_dict[variable_name] = (d_string, dist_hyper_dists)

        else:
            raw_param_dict[variable_name] = d_string

    raw_param_dict = {
        param_name: d_string
        for param_name, d_string in raw_param_dict.items()
        if param_name not in shock_hyper_params
    }

    return raw_param_dict, raw_shock_dict


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
        [parsed_result] = dist_syntax.parse_string(d_string, parse_all=True)
    except ParseException as e:
        raise InvalidDistributionException(variable_name, d_string) from e

    dist_name = parsed_result["dist_name"]
    dist_kwargs = parsed_result["dist_kwargs"]

    if dist_name not in PRELIZ_DISTS:
        raise InvalidDistributionException(variable_name, d_string)

    dist_kwargs["initial_value"] = evaluate_expression(parsed_result["initial_value"])
    dist_info = (dist_name, dist_kwargs)

    wrapper_name = parsed_result["wrapper_name"]
    wrapper_kwargs = parsed_result["wrapper_kwargs"]

    if wrapper_name is None:
        wrapper_kwargs = {}
        wrapper_info = (wrapper_name, wrapper_kwargs)
        return dist_info, wrapper_info

    if wrapper_name not in PRELIZ_DIST_WRAPPERS:
        raise ValueError(
            f'Unknown distribution wrapper {wrapper_name}. Valid functions are '
            f'{", ".join(PRELIZ_DIST_WRAPPERS.keys())} '
        )

    return (dist_name, dist_kwargs), (wrapper_name, wrapper_kwargs)


def create_preliz_distribution(variable_name, d_string):
    dist_info, wrapper_info = preprocess_distribution_string(variable_name, d_string)
    dist_name, dist_kwargs = dist_info
    wrapper_name, wrapper_kwargs = wrapper_info

    dist_kwargs.pop("initial_value", None)

    dist = getattr(pz, dist_name)(**dist_kwargs)
    if wrapper_name is not None:
        if wrapper_name == "maxent":
            wrapper_kwargs["plot"] = False
        dist = getattr(pz, wrapper_name)(dist, **wrapper_kwargs)

    return dist


class CompositeDistribution:
    def __init__(
        self,
        name: str,
        dist_name: str,
        fixed_params: dict[str, float | int],
        hyper_param_dict: dict[str, Distribution],
        param_name_to_hyper_name: dict[str, str],
    ):
        # TODO: Make this nicer
        self.name = name
        self.dist_name = dist_name
        self.hyper_param_dict = hyper_param_dict
        self.param_name_to_hyper_name = param_name_to_hyper_name
        self.fixed_params = fixed_params

    def to_pymc(self, **kwargs):
        for name, param_dist in self.hyper_param_dict.items():
            param_dist.to_pymc(name=self.param_name_to_hyper_name[name], **kwargs)


def create_composite_distribution(variable_name, outer_dist, hyper_param_dict):
    (dist_name, dist_kwargs), (wrapper_name, wrapper_kwargs) = (
        preprocess_distribution_string(variable_name, outer_dist)
    )
    if wrapper_name is not None:
        raise NotImplementedError(
            "Wrapper functions are not allowed on shock distributions"
        )
    if dist_name != "Normal":
        raise NotImplementedError(
            "Only Normal distributions are currently allowed on shocks"
        )

    initial_value = dist_kwargs.pop("initial_value", None)
    if initial_value is not None:
        raise ValueError(
            f"Initial value not allowed on shock distributions. Found initial value: {initial_value} "
            f"associated with {variable_name}"
        )

    pz_hyper_params = {}
    param_name_to_hyper_name = {}
    fixed_params = {}

    for param_name, (hyper_name, d_string) in hyper_param_dict.items():
        if hyper_name is None:
            if param_name == "mu" and d_string != 0.0:
                raise NotImplementedError(
                    f"Currently, the mean of all shocks must be zero. Found mu = {d_string} "
                    f"associated with variable {variable_name}."
                )
            fixed_params[param_name] = d_string
            continue

        if param_name == "mu":
            raise NotImplementedError(
                "Currently, only shock variance parameters can be assigned hyper-priors."
            )

        pz_dist = create_preliz_distribution(param_name, d_string)
        pz_hyper_params[param_name] = pz_dist
        param_name_to_hyper_name[param_name] = hyper_name

    composite_dist = CompositeDistribution(
        name=variable_name,
        dist_name=outer_dist,
        param_name_to_hyper_name=param_name_to_hyper_name,
        hyper_param_dict=pz_hyper_params,
        fixed_params=fixed_params,
    )

    return composite_dist


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
    raw_prior_dict: dict[str, str],
) -> tuple[
    SymbolDictionary[str, Distribution], SymbolDictionary[str, CompositeDistribution]
]:
    """
    Parameters
    ----------
    raw_prior_dict: dict[str, str]
        Dictionary of variable name: distribution string pairs.

    Returns
    -------
    param_priors: SymbolDictionary
        A dictionary of variable name: distribution pairs.
    shock_priors: dist[str, dist[str, Distribution]]
        Dictionary of shock variable names to distributions associated with their parameters.
    """
    param_priors = SymbolDictionary()
    shock_priors = SymbolDictionary()

    raw_param_dict, raw_shock_dict = split_prior_dict_by_params_and_shocks(
        raw_prior_dict
    )

    for variable_name, d_string in raw_param_dict.items():
        param_priors[variable_name] = create_preliz_distribution(
            variable_name, d_string
        )

    for variable_name, (outer_dist, hyper_param_dict) in raw_shock_dict.items():
        clean_name = variable_name[:-2]  # remove trailing []
        shock_priors[clean_name] = create_composite_distribution(
            clean_name, outer_dist, hyper_param_dict
        )

    return param_priors, shock_priors
