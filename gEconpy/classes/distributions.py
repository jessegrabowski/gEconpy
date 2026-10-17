from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from preliz.distributions.distributions import Distribution


class CompositeDistribution:
    """
    A distribution with hyper-parameters that are themselves distributions.

    Used for shock distributions where variance parameters have priors.

    Parameters
    ----------
    name : str
        Name of the variable the distribution belongs to.
    dist_name : str
        Name of the distribution family, as written in the GCN file.
    fixed_params : dict mapping str to float
        Parameters of the distribution that are given a fixed numeric value.
    hyper_param_dict : dict mapping str to Distribution
        Prior distribution for each parameter that is itself estimated, keyed by parameter name.
    param_name_to_hyper_name : dict mapping str to str
        Maps each parameter name in ``hyper_param_dict`` to the name its prior is registered under.
    """

    def __init__(
        self,
        name: str,
        dist_name: str,
        fixed_params: dict[str, float | int],
        hyper_param_dict: dict[str, "Distribution"],
        param_name_to_hyper_name: dict[str, str],
    ):
        self.name = name
        self.dist_name = dist_name
        self.hyper_param_dict = hyper_param_dict
        self.param_name_to_hyper_name = param_name_to_hyper_name
        self.fixed_params = fixed_params

    def to_pymc(self, **kwargs) -> None:
        """
        Register a PyMC random variable for every hyper-parameter prior in the current model context.

        Parameters
        ----------
        **kwargs
            Forwarded to each prior's ``to_pymc`` method.
        """
        for name, param_dist in self.hyper_param_dict.items():
            param_dist.to_pymc(name=self.param_name_to_hyper_name[name], **kwargs)


def priors_by_sampled_name(
    param_priors: Mapping[str, "Distribution"],
    shock_priors: Mapping[str, CompositeDistribution],
) -> dict[str, "Distribution"]:
    """
    Collect every prior a model declares, keyed by the name it is sampled under.

    A shock's estimated hyper-parameters, such as its standard deviation, live inside its prior. Each is keyed by
    the hyper-parameter name that :meth:`CompositeDistribution.to_pymc` registers it under.

    Parameters
    ----------
    param_priors : mapping of str to Distribution
        Prior of each model parameter, keyed by parameter name.
    shock_priors : mapping of str to CompositeDistribution
        Prior of each shock, keyed by shock name.

    Returns
    -------
    priors : dict of str to Distribution
        The parameter priors, followed by the shock hyper-parameter priors.
    """
    hyper_priors = {
        shock_prior.param_name_to_hyper_name[param_name]: hyper_prior
        for shock_prior in shock_priors.values()
        for param_name, hyper_prior in shock_prior.hyper_param_dict.items()
    }
    return {**param_priors, **hyper_priors}
