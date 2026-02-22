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


def make_lag_name(var_name: str, level: int) -> str:
    """Generate auxiliary variable name for lag chain: ``x`` -> ``x__lag1``."""
    return f"{var_name}{AUX_LAG_SEPARATOR}{level}"


def make_lead_name(var_name: str, level: int) -> str:
    """Generate auxiliary variable name for lead chain: ``x`` -> ``x__lead1``."""
    return f"{var_name}{AUX_LEAD_SEPARATOR}{level}"


class DeepTimeIndexCollector(NodeVisitor):
    """Collects variables with time indices outside [-1, 1]."""

    def __init__(self):
        self.deep_lags: dict[str, int] = {}
        self.deep_leads: dict[str, int] = {}

    def visit_Variable(self, node: Variable) -> None:
        if node.time_index.is_steady_state:
            return
        t = node.time_index.value
        if isinstance(t, int):
            if t < -1:
                self.deep_lags[node.name] = min(self.deep_lags.get(node.name, 0), t)
            elif t > 1:
                self.deep_leads[node.name] = max(self.deep_leads.get(node.name, 0), t)


class DeepTimeIndexReplacer(NodeTransformer):
    """Replaces deep time indices with references to auxiliary variables."""

    def __init__(self, deep_lags: dict[str, int], deep_leads: dict[str, int]):
        self.deep_lags = deep_lags
        self.deep_leads = deep_leads

    def visit_Variable(self, node: Variable) -> Variable:
        if node.time_index.is_steady_state:
            return node

        t = node.time_index.value
        if not isinstance(t, int):
            return node

        if node.name in self.deep_lags and t < -1:
            return Variable(
                name=make_lag_name(node.name, abs(t) - 1),
                time_index=T_MINUS_1,
                location=node.location,
            )

        if node.name in self.deep_leads and t > 1:
            return Variable(
                name=make_lead_name(node.name, t - 1),
                time_index=T_PLUS_1,
                location=node.location,
            )

        return node


def _collect_from_equations(
    equations: list[GCNEquation],
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (deep_lags, deep_leads) dicts mapping var names to extreme indices."""
    collector = DeepTimeIndexCollector()
    for eq in equations:
        collector.visit(eq)
    return collector.deep_lags, collector.deep_leads


def _create_lag_chain(var_name: str, depth: int) -> list[GCNEquation]:
    """
    Create identity equations for a backward-looking auxiliary variable chain.

    For ``x[-4]`` (depth=3), generates::

        x__lag1[] = x[-1]
        x__lag2[] = x__lag1[-1]
        x__lag3[] = x__lag2[-1]
    """
    equations = []
    for level in range(1, depth + 1):
        lag_name = make_lag_name(var_name, level)
        prev_name = var_name if level == 1 else make_lag_name(var_name, level - 1)
        equations.append(
            GCNEquation(
                lhs=Variable(name=lag_name, time_index=T),
                rhs=Variable(name=prev_name, time_index=T_MINUS_1),
            )
        )
    return equations


def _create_lead_chain(var_name: str, depth: int) -> list[GCNEquation]:
    """
    Create identity equations for a forward-looking auxiliary variable chain.

    For ``x[3]`` (depth=2), generates::

        x__lead1[] = E[][x[1]]
        x__lead2[] = E[][x__lead1[1]]

    Expectation wrapping is valid by the law of iterated expectations.
    """
    equations = []
    for level in range(1, depth + 1):
        lead_name = make_lead_name(var_name, level)
        prev_name = var_name if level == 1 else make_lead_name(var_name, level - 1)
        equations.append(
            GCNEquation(
                lhs=Variable(name=lead_name, time_index=T),
                rhs=Expectation(expr=Variable(name=prev_name, time_index=T_PLUS_1)),
            )
        )
    return equations


def expand_block_time_indices(block: GCNBlock) -> GCNBlock:
    """
    Rewrite a block so all time indices are in [-1, 0, 1].

    Variables like ``x[-4]`` are replaced with ``x__lag3[-1]``, and auxiliary
    identity equations are added to define the intermediate variables.

    Parameters
    ----------
    block : GCNBlock
        Block containing equations with arbitrary time indices.

    Returns
    -------
    GCNBlock
        Equivalent block with time indices restricted to [-1, 0, 1].
        Returns the original block unchanged if no expansion is needed.
    """
    all_equations = list(block.definitions) + list(block.objective) + list(block.constraints) + list(block.identities)
    deep_lags, deep_leads = _collect_from_equations(all_equations)

    # Also check controls for deep time indices
    control_collector = DeepTimeIndexCollector()
    for ctrl in block.controls:
        control_collector.visit(ctrl)
    for name, val in control_collector.deep_lags.items():
        deep_lags[name] = min(deep_lags.get(name, 0), val)
    for name, val in control_collector.deep_leads.items():
        deep_leads[name] = max(deep_leads.get(name, 0), val)

    if not deep_lags and not deep_leads:
        return block

    # Build auxiliary identity equations
    new_identities = list(block.identities)
    for var_name, min_lag in deep_lags.items():
        new_identities.extend(_create_lag_chain(var_name, abs(min_lag) - 1))
    for var_name, max_lead in deep_leads.items():
        new_identities.extend(_create_lead_chain(var_name, max_lead - 1))

    # Transform all references to deep indices
    transformer = DeepTimeIndexReplacer(deep_lags, deep_leads)

    def transform_equations(equations: list[GCNEquation]) -> list[GCNEquation]:
        return [transformer.visit(eq) for eq in equations]

    return GCNBlock(
        name=block.name,
        definitions=transform_equations(list(block.definitions)),
        controls=[transformer.visit(ctrl) for ctrl in block.controls],
        objective=transform_equations(list(block.objective)),
        constraints=transform_equations(list(block.constraints)),
        identities=transform_equations(new_identities),
        shocks=block.shocks,
        shock_distributions=block.shock_distributions,
        calibration=block.calibration,
        location=block.location,
    )


def expand_model_time_indices(model: GCNModel) -> GCNModel:
    """
    Rewrite all blocks in a model so time indices are in [-1, 0, 1].

    Parameters
    ----------
    model : GCNModel
        Model containing blocks with arbitrary time indices.

    Returns
    -------
    GCNModel
        Equivalent model with time indices restricted to [-1, 0, 1].
    """
    return GCNModel(
        blocks=[expand_block_time_indices(block) for block in model.blocks],
        options=model.options,
        tryreduce=model.tryreduce,
        assumptions=model.assumptions,
        filename=model.filename,
    )
