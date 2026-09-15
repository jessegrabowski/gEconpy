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


def expand_block_time_indices(block: GCNBlock) -> GCNBlock:
    """
    Rewrite a block so every time index lies in ``[-1, 0, 1]``.

    A reference such as ``x[-4]`` becomes ``x__lag3[-1]``, and identities defining the chain ``x__lag1[] = x[-1]``,
    ``x__lag2[] = x__lag1[-1]``, ``x__lag3[] = x__lag2[-1]`` are appended. Leads work the same way through
    ``x__lead<k>`` variables, with each link wrapped in an expectation.

    Parameters
    ----------
    block : GCNBlock
        Block whose equations and controls may use arbitrary time indices.

    Returns
    -------
    block : GCNBlock
        An equivalent block with time indices restricted to ``[-1, 0, 1]``. The input block itself is returned when
        it has no deep indices.
    """
    collector = DeepTimeIndexCollector()
    for node in [*block.definitions, *block.objective, *block.constraints, *block.identities, *block.controls]:
        collector.visit(node)

    deep_lags, deep_leads = collector.deep_lags, collector.deep_leads
    if not deep_lags and not deep_leads:
        return block

    auxiliary_identities = []
    for var_name, min_lag in deep_lags.items():
        auxiliary_identities.extend(_create_lag_chain(var_name, abs(min_lag) - 1))
    for var_name, max_lead in deep_leads.items():
        auxiliary_identities.extend(_create_lead_chain(var_name, max_lead - 1))

    replacer = DeepTimeIndexReplacer(deep_lags, deep_leads)

    return GCNBlock(
        name=block.name,
        definitions=[replacer.visit(eq) for eq in block.definitions],
        controls=[replacer.visit(ctrl) for ctrl in block.controls],
        objective=[replacer.visit(eq) for eq in block.objective],
        constraints=[replacer.visit(eq) for eq in block.constraints],
        identities=[replacer.visit(eq) for eq in [*block.identities, *auxiliary_identities]],
        shocks=block.shocks,
        shock_distributions=block.shock_distributions,
        calibration=block.calibration,
        location=block.location,
    )


def expand_model_time_indices(model: GCNModel) -> GCNModel:
    """
    Apply :func:`expand_block_time_indices` to every block of a model.

    Parameters
    ----------
    model : GCNModel
        Model whose blocks may use arbitrary time indices.

    Returns
    -------
    model : GCNModel
        An equivalent model with time indices restricted to ``[-1, 0, 1]``.
    """
    return GCNModel(
        blocks=[expand_block_time_indices(block) for block in model.blocks],
        options=model.options,
        tryreduce=model.tryreduce,
        assumptions=model.assumptions,
        filename=model.filename,
    )


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
    Visitor that records, per variable, the most negative lag below -1 and the largest lead above 1.

    Attributes
    ----------
    deep_lags : dict mapping str to int
        The most negative time index seen for each variable with a lag below -1.
    deep_leads : dict mapping str to int
        The largest time index seen for each variable with a lead above 1.
    """

    def __init__(self):
        self.deep_lags: dict[str, int] = {}
        self.deep_leads: dict[str, int] = {}

    def visit_Variable(self, node: Variable) -> None:
        offset = node.time_index.value
        if not isinstance(offset, int):
            return

        if offset < -1:
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
    """

    def __init__(self, deep_lags: dict[str, int], deep_leads: dict[str, int]):
        self.deep_lags = deep_lags
        self.deep_leads = deep_leads

    def visit_Variable(self, node: Variable) -> Variable:
        offset = node.time_index.value
        if not isinstance(offset, int):
            return node

        if node.name in self.deep_lags and offset < -1:
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
