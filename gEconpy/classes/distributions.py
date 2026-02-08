from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from preliz.distributions.distributions import Distribution


class CompositeDistribution:
    """
    A distribution with hyper-parameters that are themselves distributions.

    Used for shock distributions where variance parameters have priors.
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

    def to_pymc(self, **kwargs):
        for name, param_dist in self.hyper_param_dict.items():
            param_dist.to_pymc(name=self.param_name_to_hyper_name[name], **kwargs)
