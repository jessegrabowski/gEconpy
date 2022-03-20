import re

from gEcon.exceptions.exceptions import ParameterNotFoundException, \
    MultipleParameterDefinitionException, UnusedParameterError, InvalidDistributionException, \
    RepeatedParameterException, IgnoredCloseMatchWarning, UnusedParameterWarning, InvalidMeanException, \
    InvalidParameterException, DistributionOverDefinedException, InsufficientDegreesOfFreedomException

from .validation import find_typos_and_guesses

import numpy as np
from scipy.stats import rv_continuous, beta, halfnorm, gamma, invgamma, norm, uniform, truncnorm, truncexpon
from scipy import optimize
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

# Moment parameter names
MEAN_ALIASES = ['mean']
STD_ALIASES = ['std', 'sd']
MOMENTS = MEAN_ALIASES + STD_ALIASES

# Shared parameter names
LOWER_BOUND_ALIASES = ['low', 'lower', 'lower_bound', 'min']
UPPER_BOUND_ALIASES = ['high', 'upper', 'upper_bound', 'max']

# Distribution specific parameter names
NORMAL_LOC_ALIASES = ['mu', 'loc']
NORMAL_SCALE_ALIASES = ['sigma', 'tau', 'precision', 'scale']

INV_GAMMA_SHAPE_ALIASES = ['a', 'alpha', 'shape']
INV_GAMMA_SCALE_ALIASES = ['b', 'beta', 'scale']

BETA_SHAPE_ALIASES_1 = ['a', 'alpha']
BETA_SHAPE_ALIASES_2 = ['b', 'beta']

GAMMA_SHAPE_ALIASES = ['a', 'alpha', 'k', 'shape']
GAMMA_SCALE_ALIASES = ['b', 'beta', 'theta', 'scale']

DIST_ALIAS_LIST = [NORMAL_ALIASES, HALFNORMAL_ALIASES, GAMMA_ALIASES, INVERSE_GAMMA_ALIASES,
                   UNIFORM_ALIASES, BETA_ALIASES]

DIST_FUNCS = dict(zip(CANON_NAMES, [norm, halfnorm, gamma, invgamma, uniform, beta]))


class BaseDistributionParser(ABC):

    @abstractmethod
    def __init__(self, variable_name: str,
                 d_name: str,
                 loc_param_name: Optional[str],
                 scale_param_name: str,
                 shape_param_name: Optional[str],
                 all_valid_parameters: List[str]):

        self.variable_name = variable_name
        self.d_name = d_name
        self.loc_param_name = loc_param_name
        self.scale_param_name = scale_param_name
        self.shape_param_name = shape_param_name
        self.all_valid_parameters = all_valid_parameters

        self.used_parameters = []
        self.mean_constraint = None
        self.std_constraint = None

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

    @abstractmethod
    def _postprocess_parameters(self, param_dict: Dict[str, float]) -> Dict[str, float]:
        raise NotImplementedError

    def _has_moment_constraints(self):
        return self.mean_constraint is not None and self.std_constraint is not None

    def _parse_valid_required_parameter_candidates(self, canon_param_name: str,
                                                   param_names: List[str],
                                                   aliases: List[str]) -> str:

        candidates = list(set(param_names).intersection(set(aliases)))

        if len(candidates) == 0:
            # Filter all the valid parameter names before looking for typos
            invalid_param_names = list(set(param_names) - set(self.all_valid_parameters))
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
            invalid_param_names = list(set(param_names) - set(self.all_valid_parameters))

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


# TODO: Split into NormalDistributionParser and TruncatedNormalDistributionParser?
class NormalDistributionParser(BaseDistributionParser):

    def __init__(self, variable_name: str):
        super().__init__(variable_name=variable_name,
                         d_name='normal',
                         loc_param_name='loc',
                         scale_param_name='scale',
                         shape_param_name=None,
                         all_valid_parameters=NORMAL_LOC_ALIASES + NORMAL_SCALE_ALIASES + MOMENTS)

    def build_distribution(self, param_dict: Dict[str, str]) -> rv_continuous:
        parsed_param_dict = self._parse_parameters(param_dict)
        self._warn_about_unused_parameters(param_dict)

        if parsed_param_dict['a'] == -np.inf and parsed_param_dict['b'] == np.inf:
            return norm(loc=parsed_param_dict['loc'], scale=parsed_param_dict['scale'])

        if self._has_moment_constraints():
            self._verify_valid_mean_given_bounds(parsed_param_dict)
            parsed_param_dict = self._postprocess_parameters(parsed_param_dict)

        return truncnorm(**parsed_param_dict)

    def _parse_parameters(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        parsed_param_dict = {}
        parsing_functions = [self._parse_loc_parameter,
                             self._parse_scale_parameter,
                             self._parse_lower_bound,
                             self._parse_upper_bound]

        for f in parsing_functions:
            parsed_param_dict.update(f(param_dict))

        return parsed_param_dict

    def _parse_loc_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = NORMAL_LOC_ALIASES + MEAN_ALIASES
        param_names = list(param_dict.keys())
        loc_param = self._parse_valid_required_parameter_candidates('loc', param_names, aliases)
        self.used_parameters.append(loc_param)

        value = float(param_dict[loc_param])

        if loc_param in MEAN_ALIASES:
            self.mean_constraint = value

        return {self.loc_param_name: value}

    def _parse_scale_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = NORMAL_SCALE_ALIASES + STD_ALIASES
        param_names = list(param_dict.keys())
        scale_param = self._parse_valid_required_parameter_candidates('scale', param_names, aliases)
        self.used_parameters.append(scale_param)

        value = float(param_dict[scale_param])
        if scale_param in ['tau', 'precision']:
            value = 1 / value

        if scale_param in STD_ALIASES:
            self.std_constraint = value

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

    def _verify_valid_mean_given_bounds(self, param_dict: Dict[str, float]) -> None:
        loc, scale, a, b = param_dict.values()
        if not a < self.mean_constraint < b:
            raise InvalidMeanException(self.variable_name, self.d_name, loc, a, b)

    def _postprocess_parameters(self, param_dict: Dict[str, float]) -> Dict[str, float]:
        # TODO: This is way too slow. Replace the optimization with some kind of approximation? We don't know CDF(x)
        #  in closed form to get a perfect solution.
        def moment_errors(x, target_mean, target_std, a, b):
            loc_approx, scale_approx = x

            alpha = (a - loc_approx) / scale_approx
            beta = (b - loc_approx) / scale_approx
            d = truncnorm(loc=loc_approx, scale=scale_approx, a=alpha, b=beta)

            error_loc = (target_mean - d.mean())
            error_std = (target_std - d.std())

            error_vec = np.array([error_loc, error_std])

            return (error_vec ** 2).mean()

        mu, sigma, a, b = list(param_dict.values())
        result = optimize.minimize(moment_errors,
                                   x0=[mu, sigma],
                                   args=(mu, sigma, a, b),
                                   bounds=[(None, None), (0, None)],
                                   method='powell',
                                   options={'maxiter': 100})

        if not result.success and result.fun > 1e-5:
            print(result)
            raise ValueError

        loc, scale = result.x
        param_dict[self.loc_param_name] = loc
        param_dict[self.scale_param_name] = scale
        param_dict['a'] = (param_dict['a'] - loc) / scale
        param_dict['b'] = (param_dict['b'] - loc) / scale

        return param_dict


class HalfNormalDistributionParser(BaseDistributionParser):
    def __init__(self, variable_name: str):
        super().__init__(variable_name=variable_name,
                         d_name='halfnormal',
                         loc_param_name='loc',
                         scale_param_name='scale',
                         shape_param_name=None,
                         all_valid_parameters=NORMAL_SCALE_ALIASES + MOMENTS + ['loc'])

    def build_distribution(self, param_dict: Dict[str, str]) -> rv_continuous:
        parsed_param_dict = self._parse_parameters(param_dict)
        self._warn_about_unused_parameters(param_dict)

        return halfnorm(**parsed_param_dict)

    def _parse_parameters(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        parsed_param_dict = {}
        parsing_functions = [self._parse_loc_parameter,
                             self._parse_scale_parameter]

        for f in parsing_functions:
            parsed_param_dict.update(f(param_dict))

        return parsed_param_dict

    def _parse_loc_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = MEAN_ALIASES + ['loc']
        param_names = list(param_dict.keys())
        loc_param = self._parse_valid_optional_parameter_candidates('loc', param_names, aliases)
        self.used_parameters.append(loc_param)

        if loc_param is None:
            return {'loc': 0}

        value = float(param_dict[loc_param])
        return {'loc': value}

    def _parse_scale_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = NORMAL_SCALE_ALIASES + STD_ALIASES
        param_names = list(param_dict.keys())
        scale_param = self._parse_valid_required_parameter_candidates('scale', param_names, aliases)
        self.used_parameters.append(scale_param)

        value = float(param_dict[scale_param])
        if scale_param in ['tau', 'precision']:
            value = 1 / value

        if scale_param in STD_ALIASES:
            self.std_constraint = value
            value = 1

        return {self.scale_param_name: value}

    def _parse_shape_parameter(self, names: Dict[str, str]) -> Dict[str, float]:
        raise UnusedParameterError(self.d_name, 'shape')

    def _parse_lower_bound(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise UnusedParameterError(self.d_name, 'shape')

    def _parse_upper_bound(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise UnusedParameterError(self.d_name, 'upper_bound')

    def _postprocess_parameters(self, param_dict: Dict[str, float]) -> Dict[str, float]:
        used_parameters = self.used_parameters
        # HalfNormal is parameterized by only sigma, so we can't meet two moment constraints.
        if self._has_moment_constraints():
            raise InsufficientDegreesOfFreedomException(self.variable_name, self.d_name)

        if param_dict[self.scale_param_name] <= 0:
            used_name = list(set(used_parameters).intersection({'scale', 'sd', 'std'}))[0]
            raise InvalidParameterException(self.variable_name, self.d_name, 'scale', used_name, 'scale >= 0')

        if self.mean_constraint is not None:
            mu = self.mean_constraint
            param_dict[self.scale_param_name] = mu * np.sqrt(np.pi / 2)

        elif self.std_constraint is not None:
            std = self.std_constraint
            param_dict[self.scale_param_name] = np.sqrt(1 - 2 / np.pi) * std

        return param_dict


class InverseGammaDistributionParser(BaseDistributionParser):

    def __init__(self, variable_name: str):
        super().__init__(variable_name=variable_name,
                         d_name='invgamma',
                         loc_param_name='loc',
                         scale_param_name='scale',
                         shape_param_name='a',
                         all_valid_parameters=INV_GAMMA_SHAPE_ALIASES + INV_GAMMA_SCALE_ALIASES + ['loc'])

    def build_distribution(self, param_dict: Dict[str, str]) -> rv_continuous:
        parsed_param_dict = self._parse_parameters(param_dict)
        self._warn_about_unused_parameters(param_dict)
        self._postprocess_parameters(parsed_param_dict)

        return invgamma(**parsed_param_dict)

    def _parse_parameters(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        parsed_param_dict = {}
        parsing_functions = [self._parse_loc_parameter,
                             self._parse_scale_parameter,
                             self._parse_shape_parameter]

        for f in parsing_functions:
            parsed_param_dict.update(f(param_dict))

        return parsed_param_dict

    def _parse_loc_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = ['loc'] + MEAN_ALIASES
        param_names = list(param_dict.keys())
        loc_param = self._parse_valid_optional_parameter_candidates('loc', param_names, aliases)

        if loc_param is None:
            return {self.loc_param_name: 0}

        self.used_parameters.append(loc_param)
        value = float(param_dict[loc_param])

        if loc_param in MEAN_ALIASES:
            self.mean_constraint = value
            value = 0 # don't shift by the mean if its a moment condition

        return {self.loc_param_name: value}

    def _parse_scale_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = INV_GAMMA_SCALE_ALIASES + STD_ALIASES
        param_names = list(param_dict.keys())
        scale_param = self._parse_valid_required_parameter_candidates('beta', param_names, aliases)
        self.used_parameters.append(scale_param)

        value = float(param_dict[scale_param])

        if scale_param in STD_ALIASES:
            self.std_constraint = value
            value = 1 # don't scale by std if its a moment condition

        return {self.scale_param_name: value}

    def _parse_shape_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = INV_GAMMA_SHAPE_ALIASES
        param_names = list(param_dict.keys())

        # If both the mean and the std are declared we will compute the shape parameter in postprocessing
        if self._has_moment_constraints():
            shape_param = self._parse_valid_optional_parameter_candidates('alpha', param_names, aliases)
        else:
            shape_param = self._parse_valid_required_parameter_candidates('alpha', param_names, aliases)

        if shape_param is None:
            return {self.shape_param_name: 0}

        self.used_parameters.append(shape_param)
        value = float(param_dict[shape_param])
        return {self.shape_param_name: value}

    def _parse_lower_bound(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise UnusedParameterError(self.d_name, 'lower_bound')

    def _parse_upper_bound(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise UnusedParameterError(self.d_name, 'upper_bound')

    def _postprocess_parameters(self, param_dict: Dict[str, float]) -> Dict[str, float]:
        used_parameters = self.used_parameters
        user_passed_shape = any([x in used_parameters for x in INV_GAMMA_SHAPE_ALIASES])
        user_passed_scale = any([x in used_parameters for x in INV_GAMMA_SCALE_ALIASES])

        if self._has_moment_constraints():

            if user_passed_shape or user_passed_scale:
                raise DistributionOverDefinedException(self.variable_name, self.d_name, parameters=['shape (alpha)',
                                                                                                    'scale (beta)'])

            mu, std = self.mean_constraint, self.std_constraint
            if mu < 0:
                used_name = list(set(used_parameters).intersection({'mu', 'mean'}))[0]
                raise InvalidParameterException(self.variable_name, self.d_name, 'mean', used_name, 'mean >= 0')

            param_dict[self.shape_param_name] = a = (mu / std) ** 2 + 2
            param_dict[self.scale_param_name] = mu * (a - 1)

            return param_dict

        # Go through all the cases of having 1 parameter and 1 constraint to solve for the last one

        if not user_passed_shape and user_passed_scale and self.mean_constraint is not None:
            mu = self.mean_constraint
            b = param_dict[self.scale_param_name]
            param_dict[self.shape_param_name] = (b + mu) / mu

        if not user_passed_shape and user_passed_scale and self.std_constraint is not None:
            std = self.std_constraint
            b = param_dict[self.scale_param_name]

            # in principal this could be a closed form solution from the cubic formula...
            f = lambda a: b ** 2 / (a - 1) ** 2 / (a - 2) - std ** 2
            param_dict[self.shape_param_name] = optimize.root_scalar(f, bracket=[2.1, 100], method='brenth').root

        if not user_passed_scale and user_passed_shape and self.mean_constraint is not None:
            mu = self.mean_constraint
            a = param_dict[self.shape_param_name]
            param_dict[self.scale_param_name] = mu * (a - 1)

        if not user_passed_scale and user_passed_shape and self.std_constraint is not None:
            std = self.std_constraint
            a = param_dict[self.shape_param_name]
            param_dict[self.scale_param_name] = std * (a - 1) * np.sqrt((a - 2))

        return param_dict

class BetaDistributionParser(BaseDistributionParser):
    def __init__(self, variable_name: str):
        super().__init__(variable_name=variable_name,
                         d_name='beta',
                         loc_param_name='loc',
                         scale_param_name='scale',
                         shape_param_name=None,
                         all_valid_parameters=BETA_SHAPE_ALIASES_1 + BETA_SHAPE_ALIASES_2 + MOMENTS + ['loc', 'scale'])

        self.shape_param_name_1 = 'a'
        self.shape_param_name_2 = 'b'

    def build_distribution(self, param_dict: Dict[str, str]) -> rv_continuous:
        parsed_param_dict = self._parse_parameters(param_dict)
        self._warn_about_unused_parameters(param_dict)

        return halfnorm(**parsed_param_dict)

    def _parse_parameters(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        parsed_param_dict = {}
        parsing_functions = [self._parse_loc_parameter,
                             self._parse_scale_parameter,
                             self._parse_shape_parameter]

        for f in parsing_functions:
            parsed_param_dict.update(f(param_dict))

        return parsed_param_dict

    def _parse_loc_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = MEAN_ALIASES + ['loc']
        param_names = list(param_dict.keys())
        loc_param = self._parse_valid_optional_parameter_candidates('loc', param_names, aliases)
        self.used_parameters.append(loc_param)

        if loc_param is None:
            return {'loc': 0}

        value = float(param_dict[loc_param])
        if loc_param in MEAN_ALIASES:
            self.mean_constraint = value
            value = 0  # if we have a mean constraint don't shift the location

        return {'loc': value}

    def _parse_scale_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = STD_ALIASES + ['scale']
        param_names = list(param_dict.keys())
        scale_param = self._parse_valid_required_parameter_candidates('scale', param_names, aliases)
        self.used_parameters.append(scale_param)

        value = float(param_dict[scale_param])

        if scale_param in STD_ALIASES:
            self.std_constraint = value
            value = 1 # If we set a std constraint don't increase the diffusion

        return {self.scale_param_name: value}

    def _parse_shape_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        param_names = list(param_dict.keys())
        alpha_aliases = BETA_SHAPE_ALIASES_1
        beta_aliases = BETA_SHAPE_ALIASES_2

        # Check if we have moment constraints, if so, we will calculate alpha and beta in postprocessing
        if self._has_moment_constraints():
            shape_param_1 = self._parse_valid_optional_parameter_candidates('alpha', param_names, alpha_aliases)
            shape_param_2 = self._parse_valid_optional_parameter_candidates('alpha', param_names, beta_aliases)

            if not all([x is None for x in [shape_param_1, shape_param_2]]):
                raise DistributionOverDefinedException(self.variable_name, self.d_name, ['alpha', 'beta'])

        else:
            shape_param_1 = self._parse_valid_required_parameter_candidates('alpha', param_names, alpha_aliases)
            shape_param_2 = self._parse_valid_required_parameter_candidates('alpha', param_names, beta_aliases)

        alpha_value = float(param_dict[shape_param_1]) if shape_param_1 is not None else 0
        beta_value  = float(param_dict[shape_param_2]) if shape_param_2 is not None else 0

        return {self.shape_param_name_1: alpha_value,
                self.shape_param_name_2: beta_value}

    def _parse_lower_bound(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise UnusedParameterError(self.d_name, 'shape')

    def _parse_upper_bound(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise UnusedParameterError(self.d_name, 'upper_bound')

    def _postprocess_parameters(self, param_dict: Dict[str, float]) -> Dict[str, float]:
        used_parameters = self.used_parameters
        user_passed_alpha = any([x in used_parameters for x in BETA_SHAPE_ALIASES_1])
        user_passed_beta = any([x in used_parameters for x in BETA_SHAPE_ALIASES_2])

        if self._has_moment_constraints():
            mu, std = self.mean_constraint, self.std_constraint
            if not 0 < mu < 1:
                used_name = list(set(used_parameters).intersection(set(MEAN_ALIASES)))[0]
                raise InvalidParameterException(self.variable_name, self.d_name, 'mean', used_name, '0 < mean < 1')

            if not std > 0:
                used_name = list(set(used_parameters).intersection(set(STD_ALIASES)))[0]
                raise InvalidParameterException(self.variable_name, self.d_name, 'mean', used_name, 'sd > 0')

            if user_passed_alpha or user_passed_beta:
                raise DistributionOverDefinedException(self.variable_name, self.d_name, parameters=['alpha', 'beta'])

            param_dict[self.shape_param_name_1] = alpha = (-mu**3 + mu**2 - std ** 2 * mu) / std ** 2
            param_dict[self.shape_param_name_2] = alpha / mu - alpha

            return param_dict

        # Go through all cases of one parameter declaration and one constraint to compute the other parameter
        if not user_passed_alpha and user_passed_beta and self.mean_constraint is not None:
            mu = self.mean_constraint
            b = param_dict[self.shape_param_name_2]
            param_dict[self.shape_param_name_1] = (b + mu) / mu

        if not user_passed_alpha and user_passed_beta and self.std_constraint is not None:
            std = self.std_constraint
            b = param_dict[self.shape_param_name_2]
            f = lambda a: std ** 2 - a * b / (a + b) ** 2 / (a + b + 1)

            # A closed form solution exists in principle...
            param_dict[self.shape_param_name_1] = optimize.root_scalar(f, bracket=[1.001, 100], method='brenth').root

        if not user_passed_beta and user_passed_alpha and self.mean_constraint is not None:
            mu = self.mean_constraint
            a = param_dict[self.shape_param_name_1]
            param_dict[self.shape_param_name_2] = mu * (a - 1)

        if not user_passed_beta and user_passed_alpha and self.std_constraint is not None:
            std = self.std_constraint
            a = param_dict[self.shape_param_name_1]
            f = lambda b: std ** 2 - a * b / (a + b) ** 2 / (a + b + 1)
            param_dict[self.shape_param_name_2] =  optimize.root_scalar(f, bracket=[1.001, 100], method='brenth').root

        return param_dict


class GammaDistributionParser(BaseDistributionParser):

    def __init__(self, variable_name: str):
        super().__init__(variable_name=variable_name,
                         d_name='gamma',
                         loc_param_name='loc',
                         scale_param_name='scale',
                         shape_param_name='a',
                         all_valid_parameters=GAMMA_SHAPE_ALIASES + GAMMA_SCALE_ALIASES + ['loc'])

    def build_distribution(self, param_dict: Dict[str, str]) -> rv_continuous:
        parsed_param_dict = self._parse_parameters(param_dict)
        self._warn_about_unused_parameters(param_dict)
        self._postprocess_parameters(parsed_param_dict)

        return invgamma(**parsed_param_dict)

    def _parse_parameters(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        parsed_param_dict = {}
        parsing_functions = [self._parse_loc_parameter,
                             self._parse_scale_parameter,
                             self._parse_shape_parameter]

        for f in parsing_functions:
            parsed_param_dict.update(f(param_dict))

        return parsed_param_dict

    def _parse_loc_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = ['loc'] + MEAN_ALIASES
        param_names = list(param_dict.keys())
        loc_param = self._parse_valid_optional_parameter_candidates('loc', param_names, aliases)

        if loc_param is None:
            return {self.loc_param_name: 0}

        self.used_parameters.append(loc_param)
        value = float(param_dict[loc_param])

        if loc_param in MEAN_ALIASES:
            self.mean_constraint = value
            value = 0 # don't shift by the mean if its a moment condition

        return {self.loc_param_name: value}

    def _parse_scale_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = GAMMA_SCALE_ALIASES + STD_ALIASES
        param_names = list(param_dict.keys())
        scale_param = self._parse_valid_required_parameter_candidates('scale', param_names, aliases)
        self.used_parameters.append(scale_param)

        value = float(param_dict[scale_param])

        if scale_param in STD_ALIASES:
            self.std_constraint = value
            value = 1 # don't scale by std if its a moment condition

        return {self.scale_param_name: value}

    def _parse_shape_parameter(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        aliases = INV_GAMMA_SHAPE_ALIASES
        param_names = list(param_dict.keys())

        # If both the mean and the std are declared we will compute the shape parameter in postprocessing
        if self._has_moment_constraints():
            shape_param = self._parse_valid_optional_parameter_candidates('alpha', param_names, aliases)
        else:
            shape_param = self._parse_valid_required_parameter_candidates('alpha', param_names, aliases)

        if shape_param is None:
            return {self.shape_param_name: 0}

        self.used_parameters.append(shape_param)
        value = float(param_dict[shape_param])
        return {self.shape_param_name: value}

    def _parse_lower_bound(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise UnusedParameterError(self.d_name, 'lower_bound')

    def _parse_upper_bound(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        raise UnusedParameterError(self.d_name, 'upper_bound')

    def _postprocess_parameters(self, param_dict: Dict[str, float]) -> Dict[str, float]:
        used_parameters = self.used_parameters
        user_passed_shape = any([x in used_parameters for x in INV_GAMMA_SHAPE_ALIASES])
        user_passed_scale = any([x in used_parameters for x in INV_GAMMA_SCALE_ALIASES])

        if self._has_moment_constraints():

            if user_passed_shape or user_passed_scale:
                raise DistributionOverDefinedException(self.variable_name, self.d_name, parameters=['shape (alpha)',
                                                                                                    'scale (beta)'])
            mu, std = self.mean_constraint, self.std_constraint
            if mu < 0:
                used_name = list(set(used_parameters).intersection(set(MEAN_ALIASES)))[0]
                raise InvalidParameterException(self.variable_name, self.d_name, 'mean', used_name, 'mean >= 0')

            param_dict[self.shape_param_name] = a = (mu / std) ** 2
            param_dict[self.scale_param_name] = mu / a

            return param_dict

        # Go through all the cases of having 1 parameter and 1 constraint to solve for the last one

        if not user_passed_shape and user_passed_scale and self.mean_constraint is not None:
            mu = self.mean_constraint
            b = param_dict[self.scale_param_name]
            param_dict[self.shape_param_name] = mu / b

        if not user_passed_shape and user_passed_scale and self.std_constraint is not None:
            std = self.std_constraint
            b = param_dict[self.scale_param_name]
            param_dict[self.shape_param_name] = (std  / b) ** 2

        if not user_passed_scale and user_passed_shape and self.mean_constraint is not None:
            mu = self.mean_constraint
            a = param_dict[self.shape_param_name]
            param_dict[self.scale_param_name] = mu / a

        if not user_passed_scale and user_passed_shape and self.std_constraint is not None:
            std = self.std_constraint
            a = param_dict[self.shape_param_name]
            param_dict[self.scale_param_name] = std / np.sqrt(a)

        return param_dict


def build_alias_to_canon_dict(alias_list, cannon_names) -> Dict[str, str]:
    alias_to_canon_dict = {}
    aliases = reduce(lambda a, b: a + b, alias_list)

    for alias in aliases:
        for group, canon_name in zip(alias_list, cannon_names):
            if alias in group:
                alias_to_canon_dict[alias] = canon_name
                break
    return alias_to_canon_dict


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


def distribution_factory(variable_name: str, d_string: str, package: str ='scipy') -> Dict[str, int]:
    """
    Parameters
    ----------
    variable_name: str
        name of the variable with which this distribution is associated
    d_string: str
        plaintext extracted from a GCN file representing a prior distribution
    package: str
        package of the distribution function to parameterize

    Returns
    -------
    d: rv_frozen
        a scipy distribution object object

    TODO: Add options to get back pymc3 distributions
    """

    if package != 'scipy':
        raise NotImplementedError

    parser = None
    d_name, param_dict = preprocess_distribution_string(variable_name, d_string)

    if d_name == 'normal':
        parser = NormalDistributionParser(variable_name=variable_name)

    elif d_name == 'halfnormal':
        parser = HalfNormalDistributionParser(variable_name=variable_name)

    elif d_name == 'inv_gamma':
        parser = InverseGammaDistributionParser(variable_name=variable_name)

    elif d_name == 'beta':
        parser = BetaDistributionParser(variable_name=variable_name)

    elif d_name == 'gamma':
        parser = GammaDistributionParser(variable_name=variable_name)

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
