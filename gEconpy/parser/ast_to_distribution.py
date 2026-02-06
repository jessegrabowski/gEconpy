from typing import TYPE_CHECKING, Any

import preliz as pz

from preliz.distributions.distributions import Distribution

from gEconpy.exceptions import InvalidDistributionException
from gEconpy.parser.ast import GCNDistribution
from gEconpy.parser.dist_syntax import PRELIZ_DIST_WRAPPERS, PRELIZ_DISTS

if TYPE_CHECKING:
    from gEconpy.parser.ast import GCNModel


def ast_to_distribution(node: GCNDistribution) -> Distribution:
    """
    Convert a GCNDistribution AST node to a PreliZ distribution.

    Parameters
    ----------
    node : GCNDistribution
        The distribution AST node to convert.

    Returns
    -------
    Distribution
        A PreliZ distribution object.
    """
    dist_name = node.dist_name
    dist_kwargs = dict(node.dist_kwargs)

    if dist_name not in PRELIZ_DISTS:
        raise InvalidDistributionException(node.parameter_name, str(node))

    # Create the base distribution
    dist = getattr(pz, dist_name)(**dist_kwargs)

    # Apply wrapper if present
    wrapper_name = node.wrapper_name
    if wrapper_name is not None:
        if wrapper_name not in PRELIZ_DIST_WRAPPERS:
            raise ValueError(
                f"Unknown distribution wrapper {wrapper_name}. Valid wrappers are: {', '.join(PRELIZ_DIST_WRAPPERS)}"
            )

        wrapper_kwargs = dict(node.wrapper_kwargs)
        if wrapper_name == "maxent":
            wrapper_kwargs["plot"] = False

        dist = getattr(pz, wrapper_name)(dist, **wrapper_kwargs)

    return dist


def ast_to_distribution_with_metadata(
    node: GCNDistribution,
) -> tuple[Distribution, dict[str, Any]]:
    """
    Convert a GCNDistribution AST node to a PreliZ distribution with metadata.

    Parameters
    ----------
    node : GCNDistribution
        The distribution AST node to convert.

    Returns
    -------
    tuple[Distribution, dict]
        The distribution and a metadata dictionary containing:
        - parameter_name: str
        - initial_value: float | None
        - is_wrapped: bool
        - wrapper_name: str | None
    """
    dist = ast_to_distribution(node)

    metadata = {
        "parameter_name": node.parameter_name,
        "initial_value": node.initial_value,
        "is_wrapped": node.is_wrapped,
        "wrapper_name": node.wrapper_name,
    }

    return dist, metadata


def distributions_from_calibration(
    calibration_items: list,
) -> dict[str, tuple[Distribution, dict[str, Any]]]:
    """
    Extract distributions from a block's calibration list.

    Parameters
    ----------
    calibration_items : list
        The calibration list from a GCNBlock, which may contain both
        GCNEquation and GCNDistribution nodes.

    Returns
    -------
    dict[str, tuple[Distribution, dict]]
        Dictionary mapping parameter names to (distribution, metadata) tuples.
    """
    result = {}

    for item in calibration_items:
        if isinstance(item, GCNDistribution):
            dist, metadata = ast_to_distribution_with_metadata(item)
            result[item.parameter_name] = (dist, metadata)

    return result


def distributions_from_model(model: "GCNModel") -> dict[str, tuple[Distribution, dict[str, Any]]]:
    """
    Extract all distributions from a GCNModel.

    Parameters
    ----------
    model : GCNModel
        The model to extract distributions from.

    Returns
    -------
    dict[str, tuple[Distribution, dict]]
        Dictionary mapping parameter names to (distribution, metadata) tuples.
    """
    result = {}

    for block in model.blocks:
        block_dists = distributions_from_calibration(block.calibration)
        result.update(block_dists)

    return result
