import re

from gEcon.exceptions.exceptions import ParameterNotFoundException, \
    MultipleParameterDefinitionException, UnusedParameterError, InvalidDistributionException, \
    RepeatedParameterException, IgnoredCloseMatchWarning, UnusedParameterWarning

import numpy as np
from scipy.stats import rv_continuous, beta, halfnorm, gamma, invgamma, norm, uniform, truncnorm, truncexpon
from typing import Dict, Tuple, List, Optional, Union
from functools import reduce
from abc import ABC, abstractmethod
from warnings import warn

CANON_NAMES = ['normal', 'halfnormal', 'inversegamma', 'uniform', 'beta']

NORMAL_ALIASES = ['norm', 'normal', 'n']
HALFNORMAL_ALIASES = ['halfnorm', 'hn', 'halfnormal']
GAMMA_ALIASES = ['gamma', 'g']
INVERSE_GAMMA_ALIASES = ['invgamma', 'ig', 'inversegamma', 'invg', 'inverseg']
UNIFORM_ALIASES = ['u', 'uniform', 'uni', 'unif']
BETA_ALIASES = ['beta', 'b']

MEAN_ALIASES = ['mu', 'mean', 'loc']
STD_ALIASES = ['sd', 'std', 'sigma', 'scale']
PRECISION_ALIASES = ['precision', 'tau']
LOWER_BOUND_ALIASES = ['low', 'lower', 'lower_bound', 'min']
UPPER_BOUND_ALIASES = ['high', 'upper', 'upper_bound', 'max']
BETA_SHAPE_ALIASES = ['a', 'b', 'alpha', 'beta']
GAMMA_SHAPE_ALIASES = ['alpha', 'a']

DIST_ALIAS_LIST = [NORMAL_ALIASES, HALFNORMAL_ALIASES, GAMMA_ALIASES, INVERSE_GAMMA_ALIASES,
                   UNIFORM_ALIASES, BETA_ALIASES]
PARAM_ALIAS_LIST = [MEAN_ALIASES,
                    STD_ALIASES,
                    PRECISION_ALIASES,
                    LOWER_BOUND_ALIASES,
                    UPPER_BOUND_ALIASES,
                    BETA_SHAPE_ALIASES,
                    GAMMA_SHAPE_ALIASES]

ALL_PARAM_ALIASES = [alias for alias_list in PARAM_ALIAS_LIST for alias in alias_list]

DIST_FUNCS = dict(zip(CANON_NAMES, [norm, halfnorm, gamma, invgamma, uniform, beta]))


class BaseDistributionParser(ABC):

    @abstractmethod
    def __init__(self, variable_name: str, d_name: str, loc_param_name: str,
                 scale_param_name: str, shape_param_name: Optional[str]):

        self.variable_name = variable_name
        self.d_name = d_name
        self.loc_param_name = loc_param_name
        self.scale_param_name = scale_param_name
        self.shape_param_name = shape_param_name
        self.used_parameters = []

    @abstractmethod
    def build_distribution(self, param_dict: Dict[str, str]) -> rv_continuous:
        raise NotImplementedError

    @abstractmethod
    def _parse_parameters(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def _parse_loc_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def _parse_scale_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def _parse_lower_bound(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def _parse_upper_bound(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def _parse_shape_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise NotImplementedError

    def _parse_valid_required_parameter_candidates(self, canon_param_name: str,
                                                   param_names: List[str],
                                                   aliases: List[str]) -> str:

        candidates = list(set(param_names).intersection(set(aliases)))

        if len(candidates) == 0:
            # Filter all the valid parameter names before looking for typos
            invalid_param_names = list(set(param_names) - set(ALL_PARAM_ALIASES))
            best_guess, maybe_typo = None, None
            if len(invalid_param_names) > 0:
                typo_match = find_typos_and_guesses(invalid_param_names, aliases, match_threshold=0.0)
                best_guess, maybe_typo = typo_match

            raise ParameterNotFoundException(self.variable_name, self.d_name, canon_param_name, aliases,
                                             maybe_typo, best_guess)

        if len(candidates) > 1:
            raise MultipleParameterDefinitionException(self.variable_name, self.d_name, canon_param_name, candidates)

        return list(candidates)[0]

    def _parse_valid_optional_parameter_candidates(self, canon_param_name: str,
                                                   param_names: List[str],
                                                   aliases: List[str]) -> Optional[str]:
        candidates = list(set(param_names).intersection(set(aliases)))
        if len(candidates) == 0:
            invalid_param_names = list(set(param_names) - set(ALL_PARAM_ALIASES))
            if len(invalid_param_names) == 0:
                return None

            best_guess, maybe_typo = find_typos_and_guesses(invalid_param_names, aliases)
            if best_guess is not None and maybe_typo is not None:
                warn(f'Found a partial name match: "{maybe_typo}" for "{best_guess}" while parsing the {self.d_name} '
                     f'associated with "{self.variable_name}". This parameter is optional, and the partial match'
                     f'will be ignored. Please verify whether the distribution is correctly specified in the GCN file.',
                     category=IgnoredCloseMatchWarning)
                return None

        if len(candidates) > 1:
            raise MultipleParameterDefinitionException(self.variable_name, self.d_name, canon_param_name, candidates)

        return list(candidates)[0]

    def _warn_about_unused_parameters(self, param_dict: Dict[str, str]) -> None:
        used_parameters = self.used_parameters
        all_params = list(param_dict.keys())

        unused_parameters = list(set(all_params) - set(used_parameters))
        n_params = len(unused_parameters)
        if n_params > 0:
            message = f'After parsing {self.d_name} distribution associated with "{self.variable_name}", the ' \
                      f'following parameters remained unused: '
            if n_params == 1:
                message += unused_parameters[0] + '.'
            else:
                message += ', '.join(unused_parameters[:-1]) + f', and {unused_parameters[-1]}.'
            message += ' Please verify whether these parameters are needed, and adjust the GCN file accordingly.'

            warn(message, category=UnusedParameterWarning)


class NormalDistributionParser(BaseDistributionParser):

    def __init__(self, variable_name: str):
        super().__init__(variable_name=variable_name,
                         d_name='normal',
                         loc_param_name='loc',
                         scale_param_name='scale',
                         shape_param_name=None)

    def build_distribution(self, param_dict: Dict[str, str]) -> rv_continuous:
        parsed_param_dict = self._parse_parameters(param_dict)
        self._warn_about_unused_parameters(param_dict)

        if parsed_param_dict['a'] == -np.inf and parsed_param_dict['b'] == np.inf:
            return norm(loc=parsed_param_dict['loc'], scale=parsed_param_dict['scale'])

        return truncnorm(**parsed_param_dict)

    def _parse_parameters(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        parsed_param_dict = {}
        parsing_functions = [self._parse_loc_parameter,
                             self._parse_scale_parameter,
                             self._parse_upper_bound,
                             self._parse_lower_bound]

        for f in parsing_functions:
            parsed_param_dict.update(f(param_dict))

        return parsed_param_dict

    def _parse_loc_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = MEAN_ALIASES
        param_names = list(param_dict.keys())
        loc_param = self._parse_valid_required_parameter_candidates('loc', param_names, aliases)
        self.used_parameters.append(loc_param)

        value = float(param_dict[loc_param])

        return {self.loc_param_name: value}

    def _parse_scale_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = STD_ALIASES + PRECISION_ALIASES
        param_names = list(param_dict.keys())
        scale_param = self._parse_valid_required_parameter_candidates('scale', param_names, aliases)
        self.used_parameters.append(scale_param)

        value = float(param_dict[scale_param])
        if scale_param in PRECISION_ALIASES:
            value = 1 / value

        return {self.scale_param_name: value}

    def _parse_shape_parameter(self, names: Dict[str, str]) -> Dict[str, float]:
        raise UnusedParameterError(self.d_name, 'shape')

    def _parse_lower_bound(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = LOWER_BOUND_ALIASES
        param_names = list(param_dict.keys())
        lower_bound_param = self._parse_valid_optional_parameter_candidates('a', param_names, aliases)
        self.used_parameters.append(lower_bound_param)

        if lower_bound_param is None:
            return {'a': -np.inf}

        value = float(param_dict[lower_bound_param])
        return {'a': value}

    def _parse_upper_bound(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = UPPER_BOUND_ALIASES
        param_names = list(param_dict.keys())
        upper_bound_param = self._parse_valid_optional_parameter_candidates('b', param_names, aliases)
        self.used_parameters.append(upper_bound_param)

        if upper_bound_param is None:
            return {'b': np.inf}

        value = float(param_dict[upper_bound_param])
        return {'b': value}


def build_alias_to_canon_dict(alias_list, cannon_names) -> Dict[str, str]:
    alias_to_canon_dict = {}
    aliases = reduce(lambda a, b: a + b, alias_list)

    for alias in aliases:
        for group, canon_name in zip(alias_list, cannon_names):
            if alias in group:
                alias_to_canon_dict[alias] = canon_name
                break
    return alias_to_canon_dict


def jaccard_distance(s: str, d: str) -> float:
    s = set(s)
    d = set(d)
    union = len(s.union(d))
    intersection = len(s.intersection(d))

    return intersection / union


def elementwise_jaccard_distance(s: str, l: List[str]) -> List[float]:
    return [jaccard_distance(s, element) for element in l]


def find_typos_and_guesses(user_inputs: List[str],
                           valid_inputs: List[str],
                           match_threshold: float = 0.8) -> Tuple[Union[str, None], Union[str, None]]:
    # TODO: Tune match_threshold
    best_guess = max(valid_inputs,
                     key=lambda x: elementwise_jaccard_distance(x, user_inputs))
    maybe_typo = max(user_inputs,
                     key=lambda x: elementwise_jaccard_distance(x, valid_inputs))

    if jaccard_distance(best_guess, maybe_typo) < match_threshold:
        return None, None

    return best_guess, maybe_typo


def preprocess_distribution_string(variable_name: str, d_string: str) -> Tuple[str, Dict[str, str]]:
    """
    Parameters
    ----------
    variable_name: str
        A string representing the model parameter associated with this probability distribution.
    d_string: str
        A string representing a probability distribution, extracted from a GCN file by the gEcon_parser.preprocess_gcn
        function.

    Returns
    -------
    Tuple of (str, dict), containing the model parameter name associated with the distribution, and a dictionary of the
    distribution parameters (e.g. loc, scale, shape, and bounds).
    """
    dist_name_pattern = '(\w+)'

    # The not last args have a comma, while the last arg does not.
    not_last_arg_pattern = '(\w+ ?= ?\d*\.?\d*, ?)'
    last_arg_pattern = '(\w+ ?= ?\d*\.?\d* ?)'
    valid_pattern = f'{dist_name_pattern}\({not_last_arg_pattern}+?{last_arg_pattern}\)$'

    # TODO: sort out where the typo is and tell the user.
    if re.search(valid_pattern, d_string) is None:
        raise InvalidDistributionException(variable_name, d_string)

    d_name, params_string = d_string.split('(')
    params = [x.strip() for x in params_string.replace(')', '').split(',')]

    param_dict = {}
    for param in params:
        key, value = [x.strip() for x in param.split('=')]
        if key in param_dict.keys():
            raise RepeatedParameterException(variable_name, d_name, key)

        param_dict[key] = value

    return d_name, param_dict


def distribution_factory(variable_name: str,
                         param_dict: Dict[str, str],
                         distribution: str, package='scipy') -> Dict[str, int]:
    """
    Parameters
    ----------
    variable_name: str
        Name of the model variable associated with this distribution.
    param_dict: dictionary
        Dictionary of parameters:value pairs, as written in the GCN file
    distribution: str
        String indicating the name of the distribution to parameterize
    package: str
        package of the distribution function to parameterize

    Returns
    -------
    Dict
        parameter:value pairs corresponding to the parameterization of the requested package

    In general there are many ways to parameterize a distribution, and it should be possible for the user to pick
    the one that they prefer. This function tries to convert whatever parameters are written in the GCN file into
    format the distribution functions of the requested package can understand.

    Examples:
    Scipy expects parameters in loc, scale format, so:
        Input: {'mu':0, 'sigma':1}, 'normal', 'scipy'
        Output: {'loc':0, 'scale':1}

    Likewise, a user might parameterize a normal with the precision, tau, rather than sigma:
        Input: {'mu':0, 'tau':0.25}, 'normal', 'scipy'
        Output: {'loc':0, 'scale':4}

    TODO: Add options to get back pymc3 distributions
    """
    if package != 'scipy':
        raise NotImplementedError

    parser = None

    if distribution == 'normal':
        parser = NormalDistributionParser(variable_name=variable_name)

    elif distribution == 'halfnormal':
        pass

    if parser is None:
        raise ValueError('How did you even get here?')

    d = parser.build_distribution(param_dict)
    return d

#
# def distribution_factory(d_str: str, param_dict: Dict[str:int]) -> Callable:
#     d_str = d_str.lower().replace('_', '')
#     alias_to_canon_dict = build_alias_to_canon_dict(ALIAS_LIST, CANON_NAMES)
#     all_aliases = list(alias_to_canon_dict.keys())
#
#     if d_str not in all_aliases:
#         best_guess = max(all_aliases, key=lambda x: jaccard_distance(d_str, x))
#         best_guess_canon = alias_to_canon_dict[best_guess]
#
#         raise DistributionNotFoundException(d_str, best_guess, best_guess_canon)
#
#     canon_name = alias_to_canon_dict[d_str]
#     params = param_dispatcher(param_dict, canon_name)
#
#     return DIST_FUNCS[canon_name](**params)
