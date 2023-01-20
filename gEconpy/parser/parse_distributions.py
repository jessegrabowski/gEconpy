import re
from abc import ABC, abstractmethod
from functools import partial, reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
from scipy import optimize
from scipy.stats import (
    beta,
    gamma,
    halfnorm,
    invgamma,
    norm,
    rv_continuous,
    truncnorm,
    uniform,
)
from scipy.stats._distn_infrastructure import rv_frozen

from gEconpy.exceptions.exceptions import (
    DistributionOverDefinedException,
    IgnoredCloseMatchWarning,
    InsufficientDegreesOfFreedomException,
    InvalidDistributionException,
    InvalidParameterException,
    MultipleParameterDefinitionException,
    RepeatedParameterException,
    UnusedParameterWarning,
)
from gEconpy.parser.validation import find_typos_and_guesses
from gEconpy.shared.utilities import is_number

CANON_NAMES = [
    "normal",
    "truncnorm",
    "halfnormal",
    "gamma",
    "inv_gamma",
    "uniform",
    "beta",
]
NAME_TO_DIST_SCIPY_FUNC = dict(
    zip(CANON_NAMES, [norm, truncnorm, halfnorm, gamma, invgamma, uniform, beta])
)

NORMAL_ALIASES = ["norm", "normal", "n"]
TRUNCNORM_ALIASES = ["truncnorm"]
HALFNORMAL_ALIASES = ["halfnorm", "hn", "halfnormal"]
GAMMA_ALIASES = ["gamma", "g"]
INVERSE_GAMMA_ALIASES = [
    "invgamma",
    "ig",
    "inversegamma",
    "invg",
    "inverseg",
    "inv_gamma",
    "ing_g",
    "inverse_g",
    "i_g",
]
UNIFORM_ALIASES = ["u", "uniform", "uni", "unif"]
BETA_ALIASES = ["beta", "b"]

# Moment parameter names
MEAN_ALIASES = ["mean"]
STD_ALIASES = ["std", "sd"]
MOMENTS = MEAN_ALIASES + STD_ALIASES

# Shared parameter names
LOWER_BOUND_ALIASES = ["low", "lower", "lower_bound", "min"]
UPPER_BOUND_ALIASES = ["high", "upper", "upper_bound", "max"]

# Distribution specific parameter names
NORMAL_LOC_ALIASES = ["mu", "loc"]
NORMAL_SCALE_ALIASES = ["sigma", "tau", "precision", "scale"]

INV_GAMMA_SHAPE_ALIASES = ["a", "alpha", "shape"]
INV_GAMMA_SCALE_ALIASES = ["b", "beta", "scale"]

BETA_SHAPE_ALIASES_1 = ["a", "alpha"]
BETA_SHAPE_ALIASES_2 = ["b", "beta"]

GAMMA_SHAPE_ALIASES = ["a", "alpha", "k", "shape"]
GAMMA_SCALE_ALIASES = ["b", "beta", "theta", "scale"]

DIST_ALIAS_LIST = [
    NORMAL_ALIASES,
    TRUNCNORM_ALIASES,
    HALFNORMAL_ALIASES,
    GAMMA_ALIASES,
    INVERSE_GAMMA_ALIASES,
    UNIFORM_ALIASES,
    BETA_ALIASES,
]


class CompositeDistribution:
    def __init__(self, dist, **parameters):
        defined_params = {
            param: value
            for param, value in parameters.items()
            if isinstance(value, (int, float))
        }

        self.rv_params = {
            param: value
            for param, value in parameters.items()
            if isinstance(value, rv_frozen)
        }
        self.d = partial(dist, **defined_params)

    def rvs(self, size=None, random_state=None):
        sample_params = {
            param: value.rvs(size=size, random_state=random_state)
            for param, value in self.rv_params.items()
        }
        d = self.d(**sample_params)
        return d.rvs(random_state=random_state)

    def _unpack_pdf_dict(self, point_dict):
        param_dict = {
            param: value
            for param, value in point_dict.items()
            if param in self.rv_params.keys()
        }
        assert set(param_dict.keys()).union(set(self.rv_params.keys())) == set(
            self.rv_params.keys()
        )

        point_dict = {
            param: value
            for param, value in point_dict.items()
            if param not in param_dict.keys()
        }
        assert len(point_dict.keys()) == 1

        point_val = list(point_dict.values())[0]

        return param_dict, point_val

    def pdf(self, point_dict):
        pdf = 1

        param_dict, point_val = self._unpack_pdf_dict(point_dict)

        for param, value in param_dict.items():
            pdf *= self.rv_params[param].pdf(value)

        d = self.d(**param_dict)
        pdf *= d.pdf(point_val)

        return pdf

    def logpdf(self, point_dict):
        log_pdf = 0

        param_dict, point_val = self._unpack_pdf_dict(point_dict)

        for param, value in param_dict.items():
            log_pdf += self.rv_params[param].logpdf(value)

        d = self.d(**param_dict)
        log_pdf += d.logpdf(point_val)

        return log_pdf

    def conditional_rvs(self, point_dicts):
        n_samples = len(point_dicts)
        samples = np.zeros(n_samples)
        for idx in range(n_samples):
            sample_params = {
                param: value.rvs() for param, value in self.rv_params.items()
            }
            sample_params.update(point_dicts[idx])
            samples[idx] = self.d(**sample_params).rvs()

        return samples


class BaseDistributionParser(ABC):
    @abstractmethod
    def __init__(
        self,
        variable_name: str,
        d_name: str,
        loc_param_name: Optional[str],
        scale_param_name: str,
        shape_param_name: Optional[str],
        upper_bound_param_name: Optional[str],
        lower_bound_param_name: Optional[str],
        n_params: int,
        all_valid_parameters: List[str],
    ):

        self.variable_name = variable_name
        self.d_name = d_name
        self.loc_param_name = loc_param_name
        self.scale_param_name = scale_param_name
        self.shape_param_name = shape_param_name
        self.upper_bound_param_name = upper_bound_param_name
        self.lower_bound_param_name = lower_bound_param_name
        self.n_params = n_params
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

    def _parse_parameter(
        self,
        param_dict: Dict[str, str],
        aliases: List[str],
        canon_name: str,
        additional_transformation: Callable = lambda name, value: value,
    ):

        param_names = list(param_dict.keys())
        param_name = self._parse_parameter_candidates(canon_name, param_names, aliases)
        if param_name is None:
            return {}

        self.used_parameters.append(param_name)

        value = float(param_dict[param_name])
        value = additional_transformation(param_name, value)

        return {canon_name: value}

    def _verify_distribution_parameterization(self, param_dict):
        n_constraints = sum(
            [self.mean_constraint is not None, self.std_constraint is not None]
        )
        if n_constraints > self.n_params:
            raise InsufficientDegreesOfFreedomException(self.variable_name, self.d_name)

        declared_params = list(param_dict.keys())
        boundary_params = [self.lower_bound_param_name, self.upper_bound_param_name]
        declared_params = [
            param for param in declared_params if param not in boundary_params
        ]

        n_params_passed = len(declared_params)
        if (n_constraints + n_params_passed) > self.n_params:
            raise DistributionOverDefinedException(
                self.variable_name,
                self.d_name,
                self.n_params,
                n_params_passed,
                n_constraints,
            )

        if self._has_std_constraint():
            if self.std_constraint <= 0:
                raise InvalidParameterException(
                    self.variable_name,
                    self.d_name,
                    self.std_constraint,
                    self.std_constraint,
                    f"{self.std_constraint} > 0",
                )

        if (
            self.scale_param_name in declared_params
            and param_dict[self.scale_param_name] <= 0
        ):
            raise InvalidParameterException(
                self.variable_name,
                self.d_name,
                self.scale_param_name,
                self.scale_param_name,
                f"{self.scale_param_name} > 0",
            )

    @abstractmethod
    def _postprocess_parameters(self, param_dict: Dict[str, float]) -> Dict[str, float]:
        raise NotImplementedError

    def _parse_mean_constraint(self, param_dict: Dict[str, str]) -> Dict:
        aliases = MEAN_ALIASES
        param_names = list(param_dict.keys())
        mean_param = self._parse_parameter_candidates("mean", param_names, aliases)

        if mean_param is None:
            return {}

        self.used_parameters.append(mean_param)
        value = float(param_dict[mean_param])
        self.mean_constraint = value
        return {}

    def _parse_std_constraint(self, param_dict: Dict[str, str]) -> Dict:
        aliases = STD_ALIASES
        param_names = list(param_dict.keys())
        std_param = self._parse_parameter_candidates("mean", param_names, aliases)

        if std_param is None:
            return {}

        self.used_parameters.append(std_param)
        value = float(param_dict[std_param])
        self.std_constraint = value
        return {}

    def _has_mean_constraint(self):
        return self.mean_constraint is not None

    def _has_std_constraint(self):
        return self.std_constraint is not None

    def _parse_parameter_candidates(
        self, canon_param_name: str, param_names: List[str], aliases: List[str]
    ) -> Optional[str]:
        candidates = list(set(param_names).intersection(set(aliases)))
        if len(candidates) == 0:
            invalid_param_names = list(
                set(param_names) - set(self.all_valid_parameters)
            )

            if len(invalid_param_names) == 0:
                return None

            best_guess, maybe_typo = find_typos_and_guesses(
                invalid_param_names, aliases
            )

            if best_guess is not None and maybe_typo is not None:
                warn(
                    f'Found a partial name match: "{maybe_typo}" for "{best_guess}" while parsing the {self.d_name} '
                    f'associated with "{self.variable_name}". Please verify whether the distribution is correctly '
                    f"specified in the GCN file.",
                    category=IgnoredCloseMatchWarning,
                )

            return None

        if len(candidates) > 1:
            raise MultipleParameterDefinitionException(
                self.variable_name, self.d_name, canon_param_name, candidates
            )

        return list(candidates)[0]

    def _warn_about_unused_parameters(self, param_dict: Dict[str, str]) -> None:
        used_parameters = self.used_parameters
        all_params = list(param_dict.keys())

        unused_parameters = list(set(all_params) - set(used_parameters))
        n_params = len(unused_parameters)
        if n_params > 0:
            message = (
                f'After parsing {self.d_name} distribution associated with "{self.variable_name}", the '
                f"following parameters remained unused: "
            )
            if n_params == 1:
                message += unused_parameters[0] + "."
            else:
                message += (
                    ", ".join(unused_parameters[:-1])
                    + f", and {unused_parameters[-1]}."
                )
            message += " Please verify whether these parameters are needed, and adjust the GCN file accordingly."

            warn(message, category=UnusedParameterWarning)


# TODO: Split into NormalDistributionParser and TruncatedNormalDistributionParser?
class NormalDistributionParser(BaseDistributionParser):
    def __init__(self, variable_name: str):
        super().__init__(
            variable_name=variable_name,
            d_name="normal",
            loc_param_name="loc",
            scale_param_name="scale",
            shape_param_name=None,
            lower_bound_param_name="a",
            upper_bound_param_name="b",
            n_params=2,
            all_valid_parameters=NORMAL_LOC_ALIASES + NORMAL_SCALE_ALIASES + MOMENTS,
        )

    def build_distribution(
        self, param_dict: Dict[str, str], package="scipy", model=None
    ) -> rv_continuous:
        parsed_param_dict = self._parse_parameters(param_dict)

        self._warn_about_unused_parameters(param_dict)
        self._verify_distribution_parameterization(parsed_param_dict)

        if package == "scipy":
            parsed_param_dict = self._postprocess_parameters(parsed_param_dict)
            parameters = list(parsed_param_dict.keys())

            if (
                self.lower_bound_param_name not in parameters
                and self.upper_bound_param_name not in parameters
            ):
                return norm(**parsed_param_dict)
            else:
                return truncnorm(**parsed_param_dict)

    def _parse_parameters(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        parsed_param_dict = {}

        def tau_to_sigma(name, value):
            return 1 / value if name in ["tau", "precision"] else value

        parse_loc_parameter = partial(
            self._parse_parameter,
            canon_name=self.loc_param_name,
            aliases=NORMAL_LOC_ALIASES,
        )
        parse_scale_parameter = partial(
            self._parse_parameter,
            canon_name=self.scale_param_name,
            aliases=NORMAL_SCALE_ALIASES,
            additional_transformation=tau_to_sigma,
        )
        parse_lower_bound = partial(
            self._parse_parameter,
            canon_name=self.lower_bound_param_name,
            aliases=LOWER_BOUND_ALIASES,
        )
        parse_upper_bound = partial(
            self._parse_parameter,
            canon_name=self.upper_bound_param_name,
            aliases=UPPER_BOUND_ALIASES,
        )

        parsing_functions = [
            self._parse_mean_constraint,
            self._parse_std_constraint,
            parse_loc_parameter,
            parse_scale_parameter,
            parse_lower_bound,
            parse_upper_bound,
        ]

        for f in parsing_functions:
            parsed_param_dict.update(f(param_dict))

        return parsed_param_dict

    def _postprocess_parameters(self, param_dict: Dict[str, float]) -> Dict[str, float]:

        parameters = list(param_dict.keys())

        # Easiest case: No bounds
        if (
            self.lower_bound_param_name not in parameters
            and self.upper_bound_param_name not in parameters
        ):
            if self._has_mean_constraint():
                param_dict[self.loc_param_name] = self.mean_constraint
            if self._has_std_constraint():
                param_dict[self.scale_param_name] = self.std_constraint

            return param_dict

        # Handle cases with bounds
        else:
            # User might pass only one boundary, in that case set the other to infinity
            a = (
                param_dict[self.lower_bound_param_name]
                if self.lower_bound_param_name in parameters
                else -np.inf
            )
            b = (
                param_dict[self.upper_bound_param_name]
                if self.upper_bound_param_name in parameters
                else np.inf
            )

            # Case 1: Both mean and std constraint given
            if self._has_std_constraint() and self._has_mean_constraint():
                # TODO: This is extremely slow when the boundary is "binding" (i.e. the mean or std is close to it),
                #  so the loc or scale needs to become very large to meet the moment constraint. Can it be replaced
                #  with an approximation?
                def moment_errors(x, target_mean, target_std, a, b):
                    loc_approx, scale_approx = x

                    alpha = (a - loc_approx) / scale_approx
                    beta = (b - loc_approx) / scale_approx
                    d = truncnorm(loc=loc_approx, scale=scale_approx, a=alpha, b=beta)

                    error_loc = target_mean - d.mean()
                    error_std = target_std - d.std()

                    error_vec = np.array([error_loc, error_std])

                    return (error_vec**2).mean()

                mean, std = self.mean_constraint, self.std_constraint

                result = optimize.minimize(
                    moment_errors,
                    x0=[mean, std],
                    args=(mean, std, a, b),
                    bounds=[(None, None), (0, None)],
                    method="Nelder-Mead",
                    options={"maxiter": 1000},
                )

                if not result.success and result.fun > 1e-5:
                    print(result)
                    raise ValueError

                loc, scale = result.x
                param_dict[self.loc_param_name] = loc
                param_dict[self.scale_param_name] = scale

            # Case 2: Mean constraint and scale parameter
            elif self._has_mean_constraint():
                mean = self.mean_constraint
                scale = param_dict[self.scale_param_name]

                def match_mean(loc, target_mean, scale, a, b):
                    alpha = (a - loc) / scale
                    beta = (b - loc) / scale

                    return (
                        truncnorm(loc=loc, scale=scale, a=alpha, b=beta).mean()
                        - target_mean
                    )

                loc = optimize.root_scalar(
                    match_mean,
                    args=(mean, scale, a, b),
                    bracket=[-100, 100],
                    method="brenth",
                ).root

                param_dict[self.loc_param_name] = loc

            # Case 3: Scale constraint and loc parameter
            elif self._has_std_constraint():
                std = self.std_constraint
                loc = param_dict[self.loc_param_name]

                def match_std(scale, target_std, loc, a, b):
                    alpha = (a - loc) / scale
                    beta = (b - loc) / scale

                    return (
                        truncnorm(loc=loc, scale=scale, a=alpha, b=beta).std()
                        - target_std
                    )

                scale = optimize.root_scalar(
                    match_std,
                    args=(std, loc, a, b),
                    bracket=[1e-4, 100],
                    method="brenth",
                ).root

                param_dict[self.scale_param_name] = scale

            # Case 4: a, b, loc, and scale are all provided
            else:
                loc = param_dict[self.loc_param_name]
                scale = param_dict[self.scale_param_name]

            # Clean up: compute the adjusted bounds
            param_dict[self.lower_bound_param_name] = (a - loc) / scale
            param_dict[self.upper_bound_param_name] = (b - loc) / scale

        return param_dict


class HalfNormalDistributionParser(BaseDistributionParser):
    def __init__(self, variable_name: str):
        super().__init__(
            variable_name=variable_name,
            d_name="halfnormal",
            loc_param_name="loc",
            scale_param_name="scale",
            shape_param_name=None,
            upper_bound_param_name=None,
            lower_bound_param_name=None,
            n_params=2,
            all_valid_parameters=NORMAL_SCALE_ALIASES + MOMENTS + ["loc"],
        )

    def build_distribution(
        self, param_dict: Dict[str, str], package="scipy", model=None
    ) -> rv_continuous:
        parsed_param_dict = self._parse_parameters(param_dict)
        self._warn_about_unused_parameters(param_dict)
        self._verify_distribution_parameterization(parsed_param_dict)

        if package == "scipy":
            parsed_param_dict = self._postprocess_parameters(parsed_param_dict)

            return halfnorm(**parsed_param_dict)

    def _parse_parameters(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        parsed_param_dict = {}

        def tau_to_sigma(name, value):
            return 1 / value if name in ["tau", "precision"] else value

        parse_loc_parameter = partial(
            self._parse_parameter,
            canon_name=self.loc_param_name,
            aliases=[self.loc_param_name],
        )
        parse_scale_parameter = partial(
            self._parse_parameter,
            canon_name=self.scale_param_name,
            aliases=NORMAL_SCALE_ALIASES,
            additional_transformation=tau_to_sigma,
        )

        parsing_functions = [
            self._parse_mean_constraint,
            self._parse_std_constraint,
            parse_loc_parameter,
            parse_scale_parameter,
        ]

        for f in parsing_functions:
            parsed_param_dict.update(f(param_dict))

        return parsed_param_dict

    def _postprocess_parameters(self, param_dict: Dict[str, float]) -> Dict[str, float]:
        if self._has_mean_constraint() and self._has_std_constraint():
            mean, std = self.mean_constraint, self.std_constraint

            loc, scale = match_first_two_moments(
                target_mean=mean, target_std=std, dist_object=halfnorm
            )

            param_dict[self.loc_param_name] = loc
            param_dict[self.scale_param_name] = scale

        elif self._has_mean_constraint():
            mu = self.mean_constraint
            param_dict[self.scale_param_name] = mu * np.sqrt(np.pi / 2)

        elif self._has_std_constraint():
            std = self.std_constraint
            param_dict[self.scale_param_name] = std / np.sqrt(1 - 2 / np.pi)

        return param_dict


class UniformDistributionParser(BaseDistributionParser):
    def __init__(self, variable_name: str):
        super().__init__(
            variable_name=variable_name,
            d_name="uniform",
            loc_param_name="loc",
            scale_param_name="scale",
            shape_param_name=None,
            lower_bound_param_name="a",
            upper_bound_param_name="b",
            n_params=2,
            all_valid_parameters=["loc", "scale"]
            + LOWER_BOUND_ALIASES
            + UPPER_BOUND_ALIASES
            + MOMENTS,
        )

    def build_distribution(
        self, param_dict: Dict[str, str], package="scipy", model=None
    ) -> rv_continuous:
        parsed_param_dict = self._parse_parameters(param_dict)
        self._warn_about_unused_parameters(param_dict)
        self._verify_distribution_parameterization(parsed_param_dict)

        if package == "scipy":
            parsed_param_dict = self._postprocess_parameters(parsed_param_dict)

            return uniform(**parsed_param_dict)

    def _parse_parameters(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        parse_loc_parameter = partial(
            self._parse_parameter,
            canon_name=self.loc_param_name,
            aliases=[self.loc_param_name],
        )
        parse_scale_parameter = partial(
            self._parse_parameter,
            canon_name=self.scale_param_name,
            aliases=[self.scale_param_name],
        )
        parse_lower_bound = partial(
            self._parse_parameter,
            canon_name=self.lower_bound_param_name,
            aliases=LOWER_BOUND_ALIASES,
        )
        parse_upper_bound = partial(
            self._parse_parameter,
            canon_name=self.upper_bound_param_name,
            aliases=UPPER_BOUND_ALIASES,
        )

        parsing_functions = [
            self._parse_mean_constraint,
            self._parse_std_constraint,
            parse_loc_parameter,
            parse_scale_parameter,
            parse_lower_bound,
            parse_upper_bound,
        ]

        parsed_param_dict = {}
        for f in parsing_functions:
            parsed_param_dict.update(f(param_dict))

        return parsed_param_dict

    def _postprocess_parameters(self, param_dict: Dict[str, float]) -> Dict[str, float]:

        parameters = list(param_dict.keys())

        # Case 1: Two moment constraints
        if self._has_mean_constraint() and self._has_std_constraint():
            mean, std = self.mean_constraint, self.std_constraint
            b = np.sqrt(3) * std + mean
            a = 2 * mean - b
            param_dict[self.loc_param_name] = a
            param_dict[self.scale_param_name] = b - a

        # Case 2: Mean condition only
        elif self._has_mean_constraint():
            mean = self.mean_constraint

            if self.loc_param_name in parameters:
                a = param_dict[self.loc_param_name]
                b = 2 * mean - a
                param_dict[self.scale_param_name] = b - a

            if self.scale_param_name in parameters:
                scale = param_dict[self.scale_param_name]
                param_dict[self.loc_param_name] = mean - 0.5 * scale

        # Case 3: Std condition only
        elif self._has_std_constraint():
            std = self.std_constraint

            if self.loc_param_name in parameters:
                a = param_dict[self.loc_param_name]
                b = np.sqrt(12) * std + a
                param_dict[self.scale_param_name] = b - a

            elif self.scale_param_name in parameters:
                # TODO: Determine if this case is plausible, doesn't seem so, because sigma = 12 ** (-1/2) * scale
                raise ValueError(
                    "Scale and Std are not enough to identify a Uniform distribution!"
                )

        # Case 4: User passed the bounds directly, convert to loc and scale then delete
        else:
            if self.lower_bound_param_name in parameters:
                param_dict[self.loc_param_name] = param_dict[
                    self.lower_bound_param_name
                ]
                del param_dict[self.lower_bound_param_name]
            if self.upper_bound_param_name in parameters:
                b = param_dict[self.upper_bound_param_name]
                a = param_dict[self.loc_param_name]
                param_dict[self.scale_param_name] = b - a

                del param_dict[self.upper_bound_param_name]

        return param_dict


class InverseGammaDistributionParser(BaseDistributionParser):
    def __init__(self, variable_name: str):
        super().__init__(
            variable_name=variable_name,
            d_name="inv_gamma",
            loc_param_name="loc",
            scale_param_name="scale",
            shape_param_name="a",
            upper_bound_param_name=None,
            lower_bound_param_name=None,
            n_params=3,
            all_valid_parameters=INV_GAMMA_SHAPE_ALIASES
            + INV_GAMMA_SCALE_ALIASES
            + ["loc"],
        )

    def build_distribution(
        self, param_dict: Dict[str, str], package="scipy", model=None
    ) -> rv_continuous:
        parsed_param_dict = self._parse_parameters(param_dict)
        self._warn_about_unused_parameters(param_dict)
        self._verify_distribution_parameterization(parsed_param_dict)

        if package == "scipy":
            parsed_param_dict = self._postprocess_parameters(parsed_param_dict)
            return invgamma(**parsed_param_dict)

    def _parse_parameters(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        parse_loc_parameter = partial(
            self._parse_parameter,
            canon_name=self.loc_param_name,
            aliases=[self.loc_param_name],
        )
        parse_scale_parameter = partial(
            self._parse_parameter,
            canon_name=self.scale_param_name,
            aliases=INV_GAMMA_SCALE_ALIASES,
        )
        parse_shape_parameter = partial(
            self._parse_parameter,
            canon_name=self.shape_param_name,
            aliases=INV_GAMMA_SHAPE_ALIASES,
        )

        parsing_functions = [
            self._parse_mean_constraint,
            self._parse_std_constraint,
            parse_loc_parameter,
            parse_scale_parameter,
            parse_shape_parameter,
        ]

        parsed_param_dict = {}
        for f in parsing_functions:
            parsed_param_dict.update(f(param_dict))

        return parsed_param_dict

    def _postprocess_parameters(self, param_dict: Dict[str, float]) -> Dict[str, float]:
        # TODO: What should be done with loc?
        parameters = list(param_dict.keys())

        user_passed_loc = self.loc_param_name in parameters
        user_passed_scale = self.scale_param_name in parameters
        user_passed_shape = self.shape_param_name in parameters

        # Case 1: Two Constraints
        if self._has_mean_constraint() and self._has_std_constraint():
            mean, std = self.mean_constraint, self.std_constraint

            param_dict[self.shape_param_name] = a = (mean / std) ** 2 + 2
            param_dict[self.scale_param_name] = mean * (a - 1)

            return param_dict

        # Case 2: Mean constraint only
        if self._has_mean_constraint():
            mu = self.mean_constraint

            if user_passed_shape:
                a = param_dict[self.shape_param_name]
                param_dict[self.scale_param_name] = mu * (a - 1)

            elif user_passed_scale:
                b = param_dict[self.scale_param_name]
                param_dict[self.shape_param_name] = (b + mu) / mu

        # Case 3: Std constraint only
        elif self._has_std_constraint():
            std = self.std_constraint
            if user_passed_shape:
                a = param_dict[self.shape_param_name]
                param_dict[self.scale_param_name] = std * (a - 1) * np.sqrt(a - 2)

            elif user_passed_scale:
                b = param_dict[self.scale_param_name]

                def solve_for_shape(a, target_std, b):
                    return b**2 / (a - 1) ** 2 / (a - 2) - target_std**2

                param_dict[self.shape_param_name] = optimize.root_scalar(
                    solve_for_shape,
                    args=(std, b),
                    bracket=[2 + 1e-4, 100],
                    method="brenth",
                ).root
        return param_dict


class BetaDistributionParser(BaseDistributionParser):
    def __init__(self, variable_name: str):
        super().__init__(
            variable_name=variable_name,
            d_name="beta",
            loc_param_name="loc",
            scale_param_name="scale",
            shape_param_name=None,
            upper_bound_param_name=None,
            lower_bound_param_name=None,
            n_params=2,
            all_valid_parameters=BETA_SHAPE_ALIASES_1
            + BETA_SHAPE_ALIASES_2
            + MOMENTS
            + ["loc", "scale"],
        )

        self.shape_param_name_1 = "a"
        self.shape_param_name_2 = "b"

    def build_distribution(
        self, param_dict: Dict[str, str], package="scipy", model=None
    ) -> rv_continuous:
        parsed_param_dict = self._parse_parameters(param_dict)
        self._warn_about_unused_parameters(param_dict)
        self._verify_distribution_parameterization(parsed_param_dict)

        if package == "scipy":
            parsed_param_dict = self._postprocess_parameters(parsed_param_dict)
            return beta(**parsed_param_dict)

    def _parse_parameters(self, param_dict: Dict[str, str]) -> Dict[str, float]:

        parse_loc_parameter = partial(
            self._parse_parameter,
            canon_name=self.loc_param_name,
            aliases=[self.loc_param_name],
        )
        parse_scale_parameter = partial(
            self._parse_parameter,
            canon_name=self.scale_param_name,
            aliases=[self.scale_param_name],
        )
        parse_shape_parameter_1 = partial(
            self._parse_parameter,
            canon_name=self.shape_param_name_1,
            aliases=BETA_SHAPE_ALIASES_1,
        )
        parse_shape_parameter_2 = partial(
            self._parse_parameter,
            canon_name=self.shape_param_name_2,
            aliases=BETA_SHAPE_ALIASES_2,
        )

        parsing_functions = [
            self._parse_mean_constraint,
            self._parse_std_constraint,
            parse_loc_parameter,
            parse_scale_parameter,
            parse_shape_parameter_1,
            parse_shape_parameter_2,
        ]

        parsed_param_dict = {}
        for f in parsing_functions:
            parsed_param_dict.update(f(param_dict))

        return parsed_param_dict

    def _postprocess_parameters(self, param_dict: Dict[str, float]) -> Dict[str, float]:
        parameters = list(param_dict.keys())
        used_parameters = self.used_parameters

        user_passed_alpha = self.shape_param_name_1 in parameters
        user_passed_beta = self.shape_param_name_2 in parameters

        # Case 1: Two moment constraints
        if self._has_mean_constraint() and self._has_std_constraint():
            mean, std = self.mean_constraint, self.std_constraint
            if (mean > 1) or (mean < 0):
                raise InvalidParameterException(
                    self.variable_name, self.d_name, "mean", "mean", "0 < mean < 1"
                )

            if std <= 0:
                used_name = list(set(used_parameters).intersection(set(STD_ALIASES)))[0]
                raise InvalidParameterException(
                    self.variable_name, self.d_name, "mean", used_name, "sd > 0"
                )

            if ((1 - mean) ** 2 * mean) < (std**2):
                used_name = list(set(used_parameters).intersection(set(STD_ALIASES)))[0]
                raise InvalidParameterException(
                    self.variable_name,
                    self.d_name,
                    "mean, std",
                    f"mean, {used_name}",
                    "((1 - mean) ** 2 * mean) < (std ** 2)",
                )

            x = (1 - mean) / mean
            param_dict[self.shape_param_name_1] = alpha = x / (
                std**2 * (x + 1) ** 3
            ) - (1 / (x + 1))
            param_dict[self.shape_param_name_2] = alpha * x

        # # Case 2: No moment constraints
        # elif user_passed_alpha and user_passed_beta:
        #     return param_dict

        # Case 3: Mean constraint and one shape parameter
        elif self._has_mean_constraint():
            mean = self.mean_constraint
            if user_passed_alpha:
                a = param_dict[self.shape_param_name_1]
                param_dict[self.shape_param_name_2] = mean * (a - 1)

            elif user_passed_beta:
                b = param_dict[self.shape_param_name_2]
                param_dict[self.shape_param_name_1] = (b + mean) / mean

        # Case 4: Std constraints and one shape parameter
        elif self._has_std_constraint():
            std = self.std_constraint

            if user_passed_alpha:

                def solve_for_beta(b, a, std):
                    return std**2 - a * b / (a + b) ** 2 / (a + b + 1)

                a = param_dict[self.shape_param_name_1]
                b = optimize.root_scalar(
                    solve_for_beta, args=(a, std), bracket=[1.001, 100], method="brenth"
                ).root

                param_dict[self.shape_param_name_2] = b

            elif user_passed_beta:

                def solve_for_alpha(a, b, std):
                    return std**2 - a * b / (a + b) ** 2 / (a + b + 1)

                b = param_dict[self.shape_param_name_2]
                a = optimize.root_scalar(
                    solve_for_alpha,
                    args=(b, std),
                    bracket=[1.001, 100],
                    method="brenth",
                ).root
                param_dict[self.shape_param_name_1] = a

        return param_dict


class GammaDistributionParser(BaseDistributionParser):
    def __init__(self, variable_name: str):
        super().__init__(
            variable_name=variable_name,
            d_name="gamma",
            loc_param_name="loc",
            scale_param_name="scale",
            shape_param_name="a",
            lower_bound_param_name=None,
            upper_bound_param_name=None,
            n_params=3,
            all_valid_parameters=GAMMA_SHAPE_ALIASES
            + GAMMA_SCALE_ALIASES
            + MOMENTS
            + ["loc"],
        )

    def build_distribution(
        self, param_dict: Dict[str, str], package="scipy", model=None
    ) -> rv_continuous:
        parsed_param_dict = self._parse_parameters(param_dict)
        self._warn_about_unused_parameters(param_dict)
        self._verify_distribution_parameterization(parsed_param_dict)

        if package == "scipy":
            parsed_param_dict = self._postprocess_parameters(parsed_param_dict)
            return gamma(**parsed_param_dict)

    def _parse_parameters(self, param_dict: Dict[str, str]) -> Dict[str, float]:
        parse_loc_parameter = partial(
            self._parse_parameter,
            canon_name=self.loc_param_name,
            aliases=[self.loc_param_name],
        )
        parse_scale_parameter = partial(
            self._parse_parameter,
            canon_name=self.scale_param_name,
            aliases=GAMMA_SCALE_ALIASES,
        )
        parse_shape_parameter = partial(
            self._parse_parameter,
            canon_name=self.shape_param_name,
            aliases=GAMMA_SHAPE_ALIASES,
        )

        parsing_functions = [
            self._parse_mean_constraint,
            self._parse_std_constraint,
            parse_loc_parameter,
            parse_scale_parameter,
            parse_shape_parameter,
        ]

        parsed_param_dict = {}
        for f in parsing_functions:
            parsed_param_dict.update(f(param_dict))

        return parsed_param_dict

    def _postprocess_parameters(self, param_dict: Dict[str, float]) -> Dict[str, float]:
        parameters = list(param_dict.keys())

        user_passed_scale = self.scale_param_name in parameters
        user_passed_shape = self.shape_param_name in parameters

        if self._has_mean_constraint() and self._has_std_constraint():
            mean, std = self.mean_constraint, self.std_constraint
            if mean < 0:
                raise InvalidParameterException(
                    self.variable_name, self.d_name, "mean", "mean", "mean >= 0"
                )
            if std <= 0:
                raise InvalidParameterException(
                    self.variable_name, self.d_name, "std", "std", "std >= 0"
                )

            param_dict[self.shape_param_name] = a = (mean / std) ** 2
            param_dict[self.scale_param_name] = mean / a

        elif self._has_mean_constraint():
            mean = self.mean_constraint

            if user_passed_scale:
                b = param_dict[self.scale_param_name]
                param_dict[self.shape_param_name] = mean / b

            elif user_passed_shape:
                a = param_dict[self.shape_param_name]
                param_dict[self.scale_param_name] = mean / a

        elif self._has_std_constraint():
            std = self.std_constraint

            if user_passed_scale:
                b = param_dict[self.scale_param_name]
                param_dict[self.shape_param_name] = (std / b) ** 2

            elif user_passed_shape:
                a = param_dict[self.shape_param_name]
                param_dict[self.scale_param_name] = std / np.sqrt(a)

        return param_dict


def match_first_two_moments(
    target_mean: float, target_std: float, dist_object: rv_continuous
) -> Tuple[float, float]:
    def moment_errors(
        x, target_mean: float, target_std: float, dist_object: rv_continuous
    ) -> float:
        loc_approx, scale_approx = x

        d = dist_object(loc=loc_approx, scale=scale_approx)

        error_loc = target_mean - d.mean()
        error_std = target_std - d.std()

        error_vec = np.array([error_loc, error_std])

        return (error_vec**2).mean()

    result = optimize.minimize(
        moment_errors,
        x0=[target_mean, target_std],
        args=(target_mean, target_std, dist_object),
        bounds=[(None, None), (0, None)],
        method="powell",
        options={"maxiter": 100},
    )

    if not result.success and result.fun > 1e-5:
        print(result)
        raise ValueError

    loc, scale = result.x
    return loc, scale


def build_alias_to_canon_dict(alias_list, cannon_names) -> Dict[str, str]:
    alias_to_canon_dict = {}
    aliases = reduce(lambda a, b: a + b, alias_list)

    for alias in aliases:
        for group, canon_name in zip(alias_list, cannon_names):
            if alias in group:
                alias_to_canon_dict[alias] = canon_name
                break
    return alias_to_canon_dict


def preprocess_distribution_string(
    variable_name: str, d_string: str
) -> Tuple[str, Dict[str, str]]:
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
    name_to_canon_dict = build_alias_to_canon_dict(DIST_ALIAS_LIST, CANON_NAMES)

    digit_pattern = r" ?\d*\.?\d* ?"
    general_pattern = rf" ?[\w\.]* ?"

    # The not last args have a comma, while the last arg does not.
    dist_name_pattern = r"(\w+)"
    not_last_arg_pattern = rf"(\w+ ?={general_pattern}, ?)"
    last_arg_pattern = rf"(\w+ ?={general_pattern})"
    valid_pattern = (
        rf"{dist_name_pattern}\({not_last_arg_pattern}*?{last_arg_pattern}\),?$"
    )

    # TODO: sort out where the typo is and tell the user.
    if re.search(valid_pattern, d_string) is None:
        raise InvalidDistributionException(variable_name, d_string)

    d_name, params_string = d_string.split("(")
    d_name = d_name.lower()

    if d_name not in name_to_canon_dict.keys():
        raise InvalidDistributionException(variable_name, d_string)

    params = [x.strip() for x in params_string.replace(")", "").split(",")]
    params = [x for x in params if len(x) > 0]

    new_params = []
    for p in params:
        chunks = p.split("=")
        new_p = "=".join([chunks[0].lower(), chunks[1]])
        new_params.append(new_p)

    params = new_params

    param_dict = {}
    for param in params:
        key, value = (x.strip() for x in param.split("="))
        if key in param_dict.keys():
            raise RepeatedParameterException(variable_name, d_name, key)

        param_dict[key] = value

    return name_to_canon_dict[d_name], param_dict


def preprocess_prior_dict(
    raw_prior_dict: Dict[str, str]
) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
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


def distribution_factory(
    variable_name: str,
    d_name: str,
    param_dict: Dict[str, str],
    package: str = "scipy",
    model=None,
) -> rv_continuous:
    """
    Parameters
    ----------
    variable_name: str
        name of the variable with which this distribution is associated
    d_name: str
        plaintext name of the distribution to parameterize, from the CANNON_NAMES list.
    param_dict: dict
        a dictionary of parameter: value pairs, or parameter: string pairs in the case of composite distributions
    package: str
        package of the distribution function to parameterize

    Returns
    -------
    d: rv_frozen
        a scipy distribution object object
    """

    if package not in ["scipy"]:
        raise NotImplementedError

    parser = None

    if d_name == "normal":
        parser = NormalDistributionParser(variable_name=variable_name)

    elif d_name == "halfnormal":
        parser = HalfNormalDistributionParser(variable_name=variable_name)

    elif d_name == "inv_gamma":
        parser = InverseGammaDistributionParser(variable_name=variable_name)

    elif d_name == "beta":
        parser = BetaDistributionParser(variable_name=variable_name)

    elif d_name == "gamma":
        parser = GammaDistributionParser(variable_name=variable_name)

    elif d_name == "uniform":
        parser = UniformDistributionParser(variable_name=variable_name)

    if parser is None:
        print(d_name)
        raise ValueError("How did you even get here?")

    d = parser.build_distribution(param_dict, package=package, model=model)
    return d


def rename_dict_keys_with_value_transform(
    d: Dict,
    to_rename: List[str],
    new_key: str,
    variable_name: str,
    d_name: str,
    transformation: Callable = lambda name, value: value,
) -> Dict[str, Any]:
    result = {}
    matches = [key for key in d.keys() if key in set(to_rename)]
    if len(matches) > 1:
        raise MultipleParameterDefinitionException(
            variable_name, d_name, new_key, matches
        )

    for key, value in d.items():
        if key in to_rename:
            result[new_key] = transformation(key, value)
        else:
            result[key] = value

    return result


def param_values_to_floats(param_dict: Dict):
    for param, param_value in param_dict.items():
        if isinstance(param_value, str):
            if is_number(param_value):
                param_dict[param] = float(param_value)

    return param_dict


def split_out_composite_distributions(
    variable_names: List[str], d_names: List[str], param_dicts: List[Dict[str, str]]
) -> Tuple[
    Dict[str, Tuple[str, Dict[str, str]]], Dict[str, Tuple[str, Dict[str, str]]]
]:
    basic_distributions = {}
    composite_distributions = {}

    for variable_name, d_name, param_dict in zip(variable_names, d_names, param_dicts):
        if all([is_number(x) for x in param_dict.values()]):
            basic_distributions[variable_name] = (d_name, param_dict)
        else:
            composite_distributions[variable_name] = (d_name, param_dict)

    return basic_distributions, composite_distributions


def fetch_rv_params(param_dict, model):
    return_dict = {}
    for k, v in param_dict.items():
        if isinstance(v, (float, int)):
            return_dict[k] = v
        elif isinstance(v, str):
            return_dict[k] = model[v]
        else:
            raise ValueError(
                f"Found an illegal key:value pair in prior param dict, {k}:{v}"
            )

    return return_dict


def composite_distribution_factory(
    variable_name, d_name, param_dict, package="scipy", model=None
) -> Union[CompositeDistribution, None]:
    """
    Parameters
    ----------
    variable_name: str
        Name of the variable the distribution is associated with
    d_name: str
        Name of the distribution, one of CANNON_NAMES
    param_dict: dict
        Dictionary of parameter name, parameter value pairs. Parameter values should be either scipy rv_frozen objects
        or strings that can be converted to floats.
    package: str
        Which package to use to create the distributions. Currently "scipy".

    Returns
    -------
    d: CompositeDistribution
         A wrapper around a set of scipy distributions with three methods: .rvs(), .pdf(), and .logpdf()

    TODO: This function is a huge mess of if-else statements. All of this should maybe be put into the parser classes
        to take advantage of all the parameter checking that happens there. Consider this temporary.

    TODO: Currently no checks are done on the support of the parameter to ensure it matches parameter requirements
        e.g. a > 0, b > 0 in the beta distribution.

    TODO: It might be possible to do moment matching in some limited sense. Currently the initial value for the
        parameter distributions is thrown away, could use this value to moment match? Maybe not worth it.
    """

    def tau_to_scale(key, value):
        if key in {"tau", "precision"}:
            return 1 / value
        return value

    if package == "scipy":
        base_d = NAME_TO_DIST_SCIPY_FUNC[d_name]
    else:
        raise NotImplementedError('Only package = "scipy"  is supported.')

    param_dict = param_values_to_floats(param_dict)

    # validate parameters by simple rename, error on more complicated setups (no moment constraints!)
    if d_name == "normal":
        has_upper_bound = any(
            [x in set(param_dict.keys()) for x in UPPER_BOUND_ALIASES]
        )
        has_lower_bound = any(
            [x in set(param_dict.keys()) for x in LOWER_BOUND_ALIASES]
        )

        if (has_upper_bound or has_lower_bound) and package == "scipy":
            warn(
                'Moment conditions are not supported for compound distributions, and parameters "mean" and "std" will'
                'be interpreted as "loc" and "scale". Since you have passed boundaries, the first and second moments'
                "of the truncated normal distribution will not coincide with the loc and scale parameters.",
                IgnoredCloseMatchWarning,
            )

        param_dict = rename_dict_keys_with_value_transform(
            param_dict, NORMAL_LOC_ALIASES + MEAN_ALIASES, "loc", variable_name, d_name
        )
        param_dict = rename_dict_keys_with_value_transform(
            param_dict,
            NORMAL_SCALE_ALIASES + STD_ALIASES,
            "scale",
            variable_name,
            d_name,
            transformation=tau_to_scale,
        )

        param_dict = rename_dict_keys_with_value_transform(
            param_dict, LOWER_BOUND_ALIASES, "a", variable_name, d_name
        )

        param_dict = rename_dict_keys_with_value_transform(
            param_dict, UPPER_BOUND_ALIASES, "b", variable_name, d_name
        )

    elif d_name == "halfnormal" and package == "scipy":
        if any([x in set(param_dict.keys()) for x in MEAN_ALIASES]):
            warn(
                "Moment conditions are not supported for compound distributions. If you pass a random variable as a "
                "parameter value, do not pass in mean or std.",
                IgnoredCloseMatchWarning,
            )

        param_dict = rename_dict_keys_with_value_transform(
            param_dict,
            NORMAL_SCALE_ALIASES,
            "scale",
            variable_name,
            d_name,
            transformation=tau_to_scale,
        )

    elif d_name == "inv_gamma":
        if any([x in set(param_dict.keys()) for x in MOMENTS]) and package == "scipy":
            warn(
                "Moment conditions are not supported for compound distributions. If you pass a random variable as a "
                "parameter value, do not pass in mean or std.",
                IgnoredCloseMatchWarning,
            )

        param_dict = rename_dict_keys_with_value_transform(
            param_dict, INV_GAMMA_SHAPE_ALIASES, "a", variable_name, d_name
        )

        param_dict = rename_dict_keys_with_value_transform(
            param_dict, INV_GAMMA_SCALE_ALIASES, "scale", variable_name, d_name
        )

    elif d_name == "beta":
        if any([x in set(param_dict.keys()) for x in MOMENTS]) and package == "scipy":
            warn(
                "Moment conditions are not supported for compound distributions. If you pass a random variable as a "
                "parameter value, do not pass in mean or std. These conditions will be ignored, and this may cause an"
                "an error to be raised when instantiating the distribution.",
                IgnoredCloseMatchWarning,
            )

            param_dict = rename_dict_keys_with_value_transform(
                param_dict, BETA_SHAPE_ALIASES_1, "a", variable_name, d_name
            )

            param_dict = rename_dict_keys_with_value_transform(
                param_dict, BETA_SHAPE_ALIASES_2, "b", variable_name, d_name
            )

    elif d_name == "gamma":
        if any([x in set(param_dict.keys()) for x in MOMENTS]) and package == "scipy":
            warn(
                "Moment conditions are not supported for compound distributions. If you pass a random variable as a "
                "parameter value, do not pass in mean or std. These conditions will be ignored, and this may cause an"
                "an error to be raised when instantiating the distribution.",
                IgnoredCloseMatchWarning,
            )

            param_dict = rename_dict_keys_with_value_transform(
                param_dict, BETA_SHAPE_ALIASES_1, "a", variable_name, d_name
            )

            param_dict = rename_dict_keys_with_value_transform(
                param_dict, BETA_SHAPE_ALIASES_2, "b", variable_name, d_name
            )

    if package == "scipy":
        d = CompositeDistribution(base_d, **param_dict)
        return d


def create_prior_distribution_dictionary(
    raw_prior_dict: Dict[str, str]
) -> Dict[str, Any]:
    variable_names, d_names, param_dicts = preprocess_prior_dict(raw_prior_dict)
    basic_distributions, compound_distributions = split_out_composite_distributions(
        variable_names, d_names, param_dicts
    )
    prior_dict = {}

    for variable_name, (d_name, param_dict) in basic_distributions.items():
        d = distribution_factory(
            variable_name=variable_name, d_name=d_name, param_dict=param_dict
        )
        prior_dict[variable_name] = d

    for variable_name, (d_name, param_dict) in compound_distributions.items():
        rvs_used_in_d = []
        for param, value in param_dict.items():
            if value in prior_dict.keys():
                param_dict[param] = prior_dict[value]
                rvs_used_in_d.append(value)

        d = composite_distribution_factory(variable_name, d_name, param_dict)
        prior_dict[variable_name] = d
        for rv in rvs_used_in_d:
            del prior_dict[rv]

    return prior_dict
