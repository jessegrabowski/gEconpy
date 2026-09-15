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
