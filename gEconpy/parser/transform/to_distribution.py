from typing import Any

import preliz as pz

from preliz.distributions.distributions import Distribution

from gEconpy.exceptions import InvalidDistributionException
from gEconpy.parser.ast import GCNDistribution, GCNEquation, GCNModel
from gEconpy.parser.constants import PRELIZ_DIST_WRAPPERS, PRELIZ_DISTS


def ast_to_distribution(node: GCNDistribution) -> Distribution:
    """
    Build the PreliZ distribution declared by an AST node, applying its wrapper if it has one.

    Parameters
    ----------
    node : GCNDistribution
        The distribution declaration to convert.

    Returns
    -------
    dist : Distribution
        The PreliZ distribution.
    """
    if node.dist_name not in PRELIZ_DISTS:
        raise InvalidDistributionException(node.parameter_name, str(node))

    dist = getattr(pz, node.dist_name)(**node.dist_kwargs)

    if node.wrapper_name is None:
        return dist

    if node.wrapper_name not in PRELIZ_DIST_WRAPPERS:
        raise ValueError(
            f"Unknown distribution wrapper {node.wrapper_name}. Valid wrappers are: {', '.join(PRELIZ_DIST_WRAPPERS)}"
        )

    wrapper_kwargs = dict(node.wrapper_kwargs)
    if node.wrapper_name == "maxent":
        wrapper_kwargs["plot"] = False

    return getattr(pz, node.wrapper_name)(dist, **wrapper_kwargs)


def ast_to_distribution_with_metadata(
    node: GCNDistribution,
) -> tuple[Distribution, dict[str, Any]]:
    """
    Build the PreliZ distribution declared by an AST node, together with the declaration's metadata.

    Parameters
    ----------
    node : GCNDistribution
        The distribution declaration to convert.

    Returns
    -------
    dist : Distribution
        The PreliZ distribution.
    metadata : dict
        The keys ``parameter_name``, ``initial_value``, ``is_wrapped``, and ``wrapper_name``, copied from the node.
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
    calibration_items: list[GCNEquation | GCNDistribution],
) -> dict[str, tuple[Distribution, dict[str, Any]]]:
    """
    Convert the distribution declarations in a block's calibration list, skipping the equations.

    Parameters
    ----------
    calibration_items : list of GCNEquation or GCNDistribution
        The calibration list of a :class:`~gEconpy.parser.ast.GCNBlock`.

    Returns
    -------
    distributions : dict mapping str to tuple
        For each declared parameter name, the pair returned by :func:`ast_to_distribution_with_metadata`.
    """
    return {
        item.parameter_name: ast_to_distribution_with_metadata(item)
        for item in calibration_items
        if isinstance(item, GCNDistribution)
    }


def distributions_from_model(model: GCNModel) -> dict[str, tuple[Distribution, dict[str, Any]]]:
    """
    Convert every distribution declaration in a model.

    Parameters
    ----------
    model : GCNModel
        The model whose blocks hold the declarations.

    Returns
    -------
    distributions : dict mapping str to tuple
        For each declared parameter name, the pair returned by :func:`ast_to_distribution_with_metadata`. A name
        declared in several blocks keeps the last declaration.
    """
    distributions = {}
    for block in model.blocks:
        distributions.update(distributions_from_calibration(block.calibration))
    return distributions
