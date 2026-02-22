from gEconpy.parser.ast import (
    STEADY_STATE,
    T_MINUS_1,
    T_PLUS_1,
    BinaryOp,
    Expectation,
    GCNBlock,
    GCNEquation,
    GCNModel,
    Number,
    Operator,
    Parameter,
    T,
    TimeIndex,
    Variable,
)
from gEconpy.parser.errors import ParseLocation
from gEconpy.parser.transform.expand_time_indices import (
    AUX_LAG_SEPARATOR,
    AUX_LEAD_SEPARATOR,
    DeepTimeIndexCollector,
    _create_lag_chain,
    _create_lead_chain,
    expand_block_time_indices,
    expand_model_time_indices,
    make_lag_name,
    make_lead_name,
)


def _collect(node):
    collector = DeepTimeIndexCollector()
    collector.visit(node)
    return collector.deep_lags, collector.deep_leads


class TestDeepTimeIndexCollector:
    def test_finds_deep_lags_and_leads(self):
        eq = GCNEquation(
            lhs=Variable(name="w", time_index=T),
            rhs=BinaryOp(
                left=Variable(name="x", time_index=TimeIndex(-3)),
                op=Operator.ADD,
                right=Variable(name="y", time_index=TimeIndex(2)),
            ),
        )
        lags, leads = _collect(eq)
        assert lags == {"x": -3}
        assert leads == {"y": 2}

    def test_takes_extremum_for_multiple_occurrences(self):
        eq = GCNEquation(
            lhs=Variable(name="z", time_index=T),
            rhs=BinaryOp(
                left=Variable(name="x", time_index=TimeIndex(-2)),
                op=Operator.ADD,
                right=Variable(name="x", time_index=TimeIndex(-4)),
            ),
        )
        lags, _ = _collect(eq)
        assert lags == {"x": -4}

    def test_ignores_steady_state_and_normal_indices(self):
        eq = GCNEquation(
            lhs=Variable(name="z", time_index=T),
            rhs=BinaryOp(
                left=Variable(name="a", time_index=STEADY_STATE),
                op=Operator.ADD,
                right=Variable(name="b", time_index=T_MINUS_1),
            ),
        )
        lags, leads = _collect(eq)
        assert lags == {}
        assert leads == {}


class TestCreateChains:
    def test_lag_chain_structure(self):
        equations = _create_lag_chain("S", 3)
        assert len(equations) == 3
        # S__lag1[] = S[-1]
        assert equations[0].lhs.name == "S__lag1"
        assert equations[0].rhs.name == "S"
        # S__lag2[] = S__lag1[-1]
        assert equations[1].lhs.name == "S__lag2"
        assert equations[1].rhs.name == "S__lag1"
        # S__lag3[] = S__lag2[-1]
        assert equations[2].lhs.name == "S__lag3"
        assert equations[2].rhs.name == "S__lag2"

    def test_lead_chain_has_expectations(self):
        equations = _create_lead_chain("C", 2)
        assert len(equations) == 2
        # C__lead1[] = E[][C[1]]
        assert equations[0].lhs.name == "C__lead1"
        assert isinstance(equations[0].rhs, Expectation)
        assert equations[0].rhs.expr.name == "C"
        # C__lead2[] = E[][C__lead1[1]]
        assert equations[1].rhs.expr.name == "C__lead1"


class TestExpandBlockTimeIndices:
    def test_passthrough_when_no_deep_indices(self):
        block = GCNBlock(
            name="TEST",
            identities=[
                GCNEquation(
                    lhs=Variable(name="z", time_index=T),
                    rhs=Variable(name="x", time_index=T_MINUS_1),
                )
            ],
        )
        result = expand_block_time_indices(block)
        assert result is block

    def test_expands_deep_lag(self):
        block = GCNBlock(
            name="TEST",
            constraints=[
                GCNEquation(
                    lhs=Variable(name="K", time_index=T),
                    rhs=Variable(name="S", time_index=TimeIndex(-4)),
                )
            ],
        )
        result = expand_block_time_indices(block)

        # Creates 3 lag chain equations
        assert len(result.identities) == 3
        # Constraint now uses S__lag3[-1]
        assert result.constraints[0].rhs.name == "S__lag3"
        assert result.constraints[0].rhs.time_index == T_MINUS_1

    def test_expands_deep_lead(self):
        block = GCNBlock(
            name="TEST",
            identities=[
                GCNEquation(
                    lhs=Variable(name="V", time_index=T),
                    rhs=Expectation(expr=Variable(name="C", time_index=TimeIndex(3))),
                )
            ],
        )
        result = expand_block_time_indices(block)

        # Original + 2 lead equations
        assert len(result.identities) == 3
        v_eq = next(eq for eq in result.identities if eq.lhs.name == "V")
        assert v_eq.rhs.expr.name == "C__lead2"
        assert v_eq.rhs.expr.time_index == T_PLUS_1

    def test_same_variable_with_both_deep_lag_and_lead(self):
        block = GCNBlock(
            name="TEST",
            identities=[
                GCNEquation(
                    lhs=Variable(name="z", time_index=T),
                    rhs=BinaryOp(
                        left=Variable(name="x", time_index=TimeIndex(-3)),
                        op=Operator.ADD,
                        right=Expectation(expr=Variable(name="x", time_index=TimeIndex(3))),
                    ),
                )
            ],
        )
        result = expand_block_time_indices(block)

        # Original + 2 lag + 2 lead
        assert len(result.identities) == 5
        lag_eqs = [eq for eq in result.identities if "__lag" in eq.lhs.name]
        lead_eqs = [eq for eq in result.identities if "__lead" in eq.lhs.name]
        assert len(lag_eqs) == 2
        assert len(lead_eqs) == 2

    def test_transforms_all_equation_types(self):
        block = GCNBlock(
            name="TEST",
            definitions=[
                GCNEquation(
                    lhs=Variable(name="u", time_index=T),
                    rhs=Variable(name="C", time_index=TimeIndex(-2)),
                )
            ],
            objective=[
                GCNEquation(
                    lhs=Variable(name="U", time_index=T),
                    rhs=Variable(name="V", time_index=TimeIndex(2)),
                )
            ],
        )
        result = expand_block_time_indices(block)

        assert result.definitions[0].rhs.name == "C__lag1"
        assert result.objective[0].rhs.name == "V__lead1"

    def test_preserves_source_location(self):

        loc = ParseLocation(line=5, column=10, end_line=5, end_column=15)
        block = GCNBlock(
            name="TEST",
            identities=[
                GCNEquation(
                    lhs=Variable(name="z", time_index=T),
                    rhs=Variable(name="x", time_index=TimeIndex(-2), location=loc),
                )
            ],
        )
        result = expand_block_time_indices(block)

        transformed_eq = next(e for e in result.identities if e.lhs.name == "z")
        assert transformed_eq.rhs.location == loc

    def test_transforms_controls_with_deep_indices(self):
        # Control S[-4] should become S__lag3[-1]
        block = GCNBlock(
            name="TEST",
            controls=[
                Variable(name="C", time_index=T),
                Variable(name="S", time_index=TimeIndex(-4)),
            ],
            constraints=[
                GCNEquation(
                    lhs=Variable(name="K", time_index=T),
                    rhs=Variable(name="S", time_index=TimeIndex(-4)),
                )
            ],
        )
        result = expand_block_time_indices(block)

        # Controls should be transformed
        assert len(result.controls) == 2
        assert result.controls[0].name == "C"
        assert result.controls[0].time_index == T
        assert result.controls[1].name == "S__lag3"
        assert result.controls[1].time_index == T_MINUS_1


class TestExpandModelTimeIndices:
    def test_transforms_all_blocks(self):
        model = GCNModel(
            blocks=[
                GCNBlock(
                    name="BLOCK1",
                    identities=[
                        GCNEquation(
                            lhs=Variable(name="x", time_index=T),
                            rhs=Variable(name="y", time_index=TimeIndex(-3)),
                        )
                    ],
                ),
                GCNBlock(
                    name="BLOCK2",
                    identities=[
                        GCNEquation(
                            lhs=Variable(name="z", time_index=T),
                            rhs=Variable(name="w", time_index=TimeIndex(-2)),
                        )
                    ],
                ),
            ]
        )
        result = expand_model_time_indices(model)

        assert len(result.blocks[0].identities) == 3  # orig + 2 lag
        assert len(result.blocks[1].identities) == 2  # orig + 1 lag


class TestKydlandPrescottScenario:
    def test_time_to_build_model(self):
        # K[] = (1-delta)*K[-1] + S[-4]
        # IF[] = phi1*S[-3] + phi2*S[-2] + phi3*S[-1] + phi4*S[]
        block = GCNBlock(
            name="CAPITAL_PRODUCER",
            constraints=[
                GCNEquation(
                    lhs=Variable(name="K", time_index=T),
                    rhs=BinaryOp(
                        left=BinaryOp(
                            left=BinaryOp(
                                left=Number(1),
                                op=Operator.SUB,
                                right=Parameter(name="delta"),
                            ),
                            op=Operator.MUL,
                            right=Variable(name="K", time_index=T_MINUS_1),
                        ),
                        op=Operator.ADD,
                        right=Variable(name="S", time_index=TimeIndex(-4)),
                    ),
                )
            ],
            identities=[
                GCNEquation(
                    lhs=Variable(name="IF", time_index=T),
                    rhs=BinaryOp(
                        left=BinaryOp(
                            left=BinaryOp(
                                left=BinaryOp(
                                    left=Parameter(name="phi1"),
                                    op=Operator.MUL,
                                    right=Variable(name="S", time_index=TimeIndex(-3)),
                                ),
                                op=Operator.ADD,
                                right=BinaryOp(
                                    left=Parameter(name="phi2"),
                                    op=Operator.MUL,
                                    right=Variable(name="S", time_index=TimeIndex(-2)),
                                ),
                            ),
                            op=Operator.ADD,
                            right=BinaryOp(
                                left=Parameter(name="phi3"),
                                op=Operator.MUL,
                                right=Variable(name="S", time_index=T_MINUS_1),
                            ),
                        ),
                        op=Operator.ADD,
                        right=BinaryOp(
                            left=Parameter(name="phi4"),
                            op=Operator.MUL,
                            right=Variable(name="S", time_index=T),
                        ),
                    ),
                )
            ],
        )

        result = expand_block_time_indices(block)

        # IF identity + 3 lag chain equations
        assert len(result.identities) == 4

        # Verify no deep indices remain
        def has_deep_indices(node):
            if isinstance(node, Variable) and not node.time_index.is_steady_state:
                t = node.time_index.value
                return isinstance(t, int) and (t < -1 or t > 1)
            if isinstance(node, BinaryOp):
                return has_deep_indices(node.left) or has_deep_indices(node.right)
            if isinstance(node, GCNEquation):
                return has_deep_indices(node.lhs) or has_deep_indices(node.rhs)
            return False

        for eq in result.constraints + result.identities:
            assert not has_deep_indices(eq)
