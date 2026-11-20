from collections.abc import Collection
from dataclasses import replace

from gEconpy.parser.ast import (
    T_MINUS_1,
    T_PLUS_1,
    Expectation,
    GCNBlock,
    GCNEquation,
    GCNModel,
    NodeTransformer,
    NodeVisitor,
    T,
    Variable,
)

AUX_LAG_SEPARATOR = "__lag"
AUX_LEAD_SEPARATOR = "__lead"


def expand_model_time_indices(model: GCNModel) -> GCNModel:
    """
    Rewrite every block of a model so every time index lies in ``[-1, 0, 1]`` and no shock appears lagged.

    A reference such as ``x[-4]`` becomes ``x__lag3[-1]``, and identities defining the chain ``x__lag1[] = x[-1]``,
    ``x__lag2[] = x__lag1[-1]``, ``x__lag3[] = x__lag2[-1]`` are added. Leads work the same way through
    ``x__lead<k>`` variables, with each link wrapped in an expectation. A lagged shock ``epsilon[-k]`` becomes
    ``epsilon__lag<k-1>[-1]``, and its chain starts from ``epsilon__lag0[] = epsilon[]``, so the lag is carried by a
    state variable.

    Each chain is added once for the whole model, at the deepest index any block uses. A shock's chain goes in the
    block that declares the shock, and a variable's chain goes in the first block that uses the deep index.

    Parameters
    ----------
    model : GCNModel
        Model whose blocks may use arbitrary time indices.

    Returns
    -------
    model : GCNModel
        An equivalent model with time indices restricted to ``[-1, 0, 1]``. The input model itself is returned when
        no block has deep indices or lagged shocks.
    """
    blocks = model.blocks
    shock_names = {shock.name for block in blocks for shock in block.shocks}
    deep_lags: dict[str, int] = {}
    deep_leads: dict[str, int] = {}
    shock_lags: dict[str, int] = {}
    chain_owners: dict[tuple[str, str], int] = {}

    for index, block in enumerate(blocks):
        collector = DeepTimeIndexCollector(shock_names=shock_names)
        for node in [*block.definitions, *block.objective, *block.constraints, *block.identities, *block.controls]:
            collector.visit(node)
        for found, merged, kind, extreme in [
            (collector.deep_lags, deep_lags, "lag", min),
            (collector.deep_leads, deep_leads, "lead", max),
            (collector.shock_lags, shock_lags, "shock", min),
        ]:
            for name, offset in found.items():
                merged[name] = extreme(merged.get(name, offset), offset)
                chain_owners.setdefault((kind, name), index)

    if not deep_lags and not deep_leads and not shock_lags:
        return model

    for index, block in enumerate(blocks):
        for shock in block.shocks:
            if shock.name in shock_lags:
                chain_owners["shock", shock.name] = index

    chains: list[list[GCNEquation]] = [[] for _ in blocks]
    for name, min_lag in deep_lags.items():
        chains[chain_owners["lag", name]].extend(_create_lag_chain(name, abs(min_lag) - 1))
    for name, max_lead in deep_leads.items():
        chains[chain_owners["lead", name]].extend(_create_lead_chain(name, max_lead - 1))
    for name, min_lag in shock_lags.items():
        chains[chain_owners["shock", name]].extend(_create_shock_lag_chain(name, abs(min_lag)))

    replacer = DeepTimeIndexReplacer(deep_lags, deep_leads, shock_lags)
    expanded = [
        _rewrite_block(block, replacer, block_chains) for block, block_chains in zip(blocks, chains, strict=True)
    ]
    return replace(model, blocks=expanded)


def make_lag_name(var_name: str, level: int) -> str:
    """
    Build the auxiliary variable name for one link of a lag chain, turning ``x`` at level 1 into ``x__lag1``.

    Parameters
    ----------
    var_name : str
        Name of the original variable.
    level : int
        Position in the lag chain, counting from 1.

    Returns
    -------
    name : str
        The auxiliary variable name.
    """
    return f"{var_name}{AUX_LAG_SEPARATOR}{level}"


def make_lead_name(var_name: str, level: int) -> str:
    """
    Build the auxiliary variable name for one link of a lead chain, turning ``x`` at level 1 into ``x__lead1``.

    Parameters
    ----------
    var_name : str
        Name of the original variable.
    level : int
        Position in the lead chain, counting from 1.

    Returns
    -------
    name : str
        The auxiliary variable name.
    """
    return f"{var_name}{AUX_LEAD_SEPARATOR}{level}"


class DeepTimeIndexCollector(NodeVisitor):
    """
    Visitor that records the deep lags and leads of variables and the lags of shocks.

    Parameters
    ----------
    shock_names : iterable of str
        Names of the model's shocks, whose lags are recorded from -1 down in ``shock_lags``.

    Attributes
    ----------
    deep_lags : dict mapping str to int
        The most negative time index seen for each variable with a lag below -1.
    deep_leads : dict mapping str to int
        The largest time index seen for each variable with a lead above 1.
    shock_lags : dict mapping str to int
        The most negative time index seen for each lagged shock.
    """

    def __init__(self, shock_names: Collection[str]):
        self.shock_names = shock_names
        self.deep_lags: dict[str, int] = {}
        self.deep_leads: dict[str, int] = {}
        self.shock_lags: dict[str, int] = {}

    def visit_Variable(self, node: Variable) -> None:
        offset = node.time_index.value
        if not isinstance(offset, int):
            return

        if node.name in self.shock_names and offset < 0:
            self.shock_lags[node.name] = min(self.shock_lags.get(node.name, 0), offset)
        elif offset < -1:
            self.deep_lags[node.name] = min(self.deep_lags.get(node.name, 0), offset)
        elif offset > 1:
            self.deep_leads[node.name] = max(self.deep_leads.get(node.name, 0), offset)


class DeepTimeIndexReplacer(NodeTransformer):
    """
    Transformer that rewrites deep time indices as references to auxiliary chain variables.

    Parameters
    ----------
    deep_lags : dict mapping str to int
        Variables with a lag below -1, as collected by :class:`DeepTimeIndexCollector`.
    deep_leads : dict mapping str to int
        Variables with a lead above 1, as collected by :class:`DeepTimeIndexCollector`.
    shock_lags : dict mapping str to int
        Lagged shocks, as collected by :class:`DeepTimeIndexCollector`.
    """

    def __init__(self, deep_lags: dict[str, int], deep_leads: dict[str, int], shock_lags: dict[str, int]):
        self.deep_lags = deep_lags
        self.deep_leads = deep_leads
        self.shock_lags = shock_lags

    def visit_Variable(self, node: Variable) -> Variable:
        offset = node.time_index.value
        if not isinstance(offset, int):
            return node

        is_lagged_shock = node.name in self.shock_lags and offset < 0
        if is_lagged_shock or (node.name in self.deep_lags and offset < -1):
            return Variable(
                name=make_lag_name(node.name, abs(offset) - 1), time_index=T_MINUS_1, location=node.location
            )

        if node.name in self.deep_leads and offset > 1:
            return Variable(name=make_lead_name(node.name, offset - 1), time_index=T_PLUS_1, location=node.location)

        return node


def _create_lag_chain(var_name: str, depth: int) -> list[GCNEquation]:
    equations = []
    for level in range(1, depth + 1):
        previous_name = var_name if level == 1 else make_lag_name(var_name, level - 1)
        equations.append(
            GCNEquation(
                lhs=Variable(name=make_lag_name(var_name, level), time_index=T),
                rhs=Variable(name=previous_name, time_index=T_MINUS_1),
            )
        )
    return equations


def _create_shock_lag_chain(shock_name: str, depth: int) -> list[GCNEquation]:
    links = [(make_lag_name(shock_name, 0), Variable(name=shock_name, time_index=T))]
    links.extend(
        (make_lag_name(shock_name, level), Variable(name=make_lag_name(shock_name, level - 1), time_index=T_MINUS_1))
        for level in range(1, depth)
    )
    return [GCNEquation(lhs=Variable(name=name, time_index=T), rhs=rhs) for name, rhs in links]


def _create_lead_chain(var_name: str, depth: int) -> list[GCNEquation]:
    # Each link is wrapped in an expectation, which the law of iterated expectations makes exact.
    equations = []
    for level in range(1, depth + 1):
        previous_name = var_name if level == 1 else make_lead_name(var_name, level - 1)
        equations.append(
            GCNEquation(
                lhs=Variable(name=make_lead_name(var_name, level), time_index=T),
                rhs=Expectation(expr=Variable(name=previous_name, time_index=T_PLUS_1)),
            )
        )
    return equations


def _rewrite_block(block: GCNBlock, replacer: DeepTimeIndexReplacer, chains: list[GCNEquation]) -> GCNBlock:
    return GCNBlock(
        name=block.name,
        definitions=[replacer.visit(eq) for eq in block.definitions],
        controls=[replacer.visit(ctrl) for ctrl in block.controls],
        objective=[replacer.visit(eq) for eq in block.objective],
        constraints=[replacer.visit(eq) for eq in block.constraints],
        identities=[replacer.visit(eq) for eq in [*block.identities, *chains]],
        shocks=block.shocks,
        shock_distributions=block.shock_distributions,
        calibration=block.calibration,
        location=block.location,
    )
