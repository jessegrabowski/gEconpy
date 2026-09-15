import pytest

from gEconpy.parser.ast import (
    STEADY_STATE,
    T_MINUS_1,
    T_PLUS_1,
    BinaryOp,
    Expectation,
    FunctionCall,
    GCNBlock,
    GCNDistribution,
    GCNEquation,
    GCNModel,
    Number,
    Operator,
    Parameter,
    T,
    Tag,
    TimeIndex,
    UnaryOp,
    Variable,
)
from gEconpy.parser.errors import ParseLocation

LOCATION = ParseLocation(line=3, column=7, end_line=3, end_column=12, source_line="    Y[] = C[];")


class TestTimeIndex:
    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="must be 'ss'"):
            TimeIndex("invalid")

    def test_equality_with_raw_values(self):
        assert TimeIndex(0) == 0
        assert TimeIndex("ss") == "ss"
        assert TimeIndex(1) != TimeIndex(-1)

    def test_step_from_steady_state_raises(self):
        t = TimeIndex("ss")
        with pytest.raises(ValueError):
            t.step_forward()
        with pytest.raises(ValueError):
            t.step_backward()

    def test_step_forward_and_backward(self):
        assert TimeIndex(0).step_forward() == TimeIndex(1)
        assert TimeIndex(0).step_backward() == TimeIndex(-1)
        assert TimeIndex(-1).step_forward() == TimeIndex(0)

    def test_can_be_used_as_dict_key(self):
        d = {TimeIndex(0): "now", TimeIndex(1): "future"}
        assert d[T] == "now"
        assert d[T_PLUS_1] == "future"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, "[]"), (1, "[1]"), (-1, "[-1]"), (-3, "[-3]"), ("ss", "[ss]")],
    )
    def test_str(self, value, expected):
        assert str(TimeIndex(value)) == expected


class TestTag:
    @pytest.mark.parametrize("name", ["exclude", "EXCLUDE", "Exclude"])
    def test_from_string_ignores_case(self, name):
        assert Tag.from_string(name) is Tag.EXCLUDE

    def test_from_string_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown tag '@bogus'"):
            Tag.from_string("bogus")


class TestVariable:
    def test_at_accepts_int_or_string(self):
        v = Variable(name="X")
        assert v.at(1).time_index == T_PLUS_1
        assert v.at("ss").time_index == STEADY_STATE

    def test_equality_considers_time_index(self):
        assert Variable(name="X", time_index=T) != Variable(name="X", time_index=T_PLUS_1)
        assert Variable(name="X") == Variable(name="X", time_index=TimeIndex(0))

    def test_can_be_used_in_set(self):
        variables = {
            Variable(name="C"),
            Variable(name="C"),
            Variable(name="C", time_index=T_PLUS_1),
            Variable(name="K"),
        }
        assert len(variables) == 3

    def test_at_and_to_ss_keep_location(self):
        v = Variable(name="X", location=LOCATION)
        assert v.at(-1) == Variable(name="X", time_index=T_MINUS_1)
        assert v.at(-1).location == LOCATION
        assert v.to_ss().time_index == STEADY_STATE
        assert v.to_ss().location == LOCATION


class TestWithLocation:
    @pytest.mark.parametrize(
        "node",
        [
            Number(value=2.5),
            Parameter(name="alpha"),
            Variable(name="K", time_index=T_MINUS_1),
            BinaryOp(left=Parameter(name="a"), op=Operator.POW, right=Parameter(name="b")),
            UnaryOp(op=Operator.NEG, operand=Parameter(name="a")),
            FunctionCall(func_name="log", args=(Parameter(name="a"), Parameter(name="b"))),
            Expectation(expr=Variable(name="U", time_index=T_PLUS_1)),
            GCNEquation(
                lhs=Variable(name="Y"),
                rhs=Variable(name="C"),
                lagrange_multiplier="lambda",
                calibrating_parameter="beta",
                tags=frozenset([Tag.EXCLUDE]),
            ),
            GCNDistribution(
                parameter_name="alpha",
                dist_name="Beta",
                dist_kwargs={"alpha": 2, "beta": 5},
                wrapper_name="maxent",
                wrapper_kwargs={"lower": 0.2},
                initial_value=0.35,
            ),
        ],
        ids=lambda node: type(node).__name__,
    )
    def test_copies_every_field_and_attaches_location(self, node):
        relocated = node.with_location(LOCATION)

        assert relocated == node
        assert relocated.location == LOCATION
        assert node.location is None
        assert {k: v for k, v in vars(relocated).items() if k != "location"} == {
            k: v for k, v in vars(node).items() if k != "location"
        }


class TestExpressionNodes:
    def test_nested_binary_ops_preserve_structure(self):
        left_assoc = BinaryOp(
            left=BinaryOp(left=Variable(name="C"), op=Operator.ADD, right=Variable(name="I")),
            op=Operator.MUL,
            right=Variable(name="K"),
        )
        right_assoc = BinaryOp(
            left=Variable(name="C"),
            op=Operator.ADD,
            right=BinaryOp(left=Variable(name="I"), op=Operator.MUL, right=Variable(name="K")),
        )
        assert left_assoc != right_assoc

    def test_function_call_with_nested_expression(self):
        expr = FunctionCall(
            func_name="log",
            args=(BinaryOp(left=Variable(name="C"), op=Operator.DIV, right=Variable(name="L")),),
        )
        assert str(expr) == "log((C[] / L[]))"

    def test_expectation_contains_expression(self):
        inner = BinaryOp(
            left=Parameter(name="beta"),
            op=Operator.MUL,
            right=Variable(name="U", time_index=T_PLUS_1),
        )
        e = Expectation(expr=inner)
        assert e.expr.left == Parameter(name="beta")


class TestGCNEquation:
    def test_lagrange_and_calibrating_flags(self):
        eq_lagrange = GCNEquation(
            lhs=Variable(name="Y"),
            rhs=Variable(name="C"),
            lagrange_multiplier="lambda",
        )
        eq_calib = GCNEquation(
            lhs=Variable(name="Y"),
            rhs=Parameter(name="beta"),
            calibrating_parameter="beta",
        )
        assert eq_lagrange.has_lagrange_multiplier and not eq_lagrange.is_calibrating
        assert eq_calib.is_calibrating and not eq_calib.has_lagrange_multiplier

    def test_with_tags_replaces_tags_and_keeps_everything_else(self):
        eq = GCNEquation(
            lhs=Variable(name="Y"),
            rhs=Variable(name="C"),
            lagrange_multiplier="lambda",
            tags=frozenset([Tag.EXCLUDE]),
            location=LOCATION,
        )
        retagged = eq.with_tags(frozenset([Tag.MINIMIZE]))

        assert retagged.tags == frozenset([Tag.MINIMIZE])
        assert retagged.lagrange_multiplier == "lambda"
        assert retagged.location == LOCATION
        assert eq.tags == frozenset([Tag.EXCLUDE])

    def test_str_includes_tags_multiplier_and_calibrating_parameter(self):
        eq = GCNEquation(
            lhs=Variable(name="Y"),
            rhs=Variable(name="C"),
            lagrange_multiplier="lambda",
            calibrating_parameter="beta",
            tags=frozenset([Tag.MINIMIZE, Tag.EXCLUDE]),
        )
        assert str(eq) == "@exclude\n@minimize\nY[] = C[] : lambda -> beta"


class TestGCNBlock:
    @pytest.mark.parametrize(
        ("controls", "objective", "expected"),
        [
            ([Variable(name="C")], [], False),
            ([], [GCNEquation(lhs=Variable(name="U"), rhs=Variable(name="u"))], False),
            ([Variable(name="C")], [GCNEquation(lhs=Variable(name="U"), rhs=Variable(name="u"))], True),
        ],
    )
    def test_has_optimization_problem_requires_both_controls_and_objective(self, controls, objective, expected):
        block = GCNBlock(name="TEST", controls=controls, objective=objective)
        assert block.has_optimization_problem() is expected


class TestGCNModel:
    def test_all_variables_traverses_nested_expressions(self):
        eq = GCNEquation(
            lhs=Variable(name="Y"),
            rhs=BinaryOp(
                left=Variable(name="C"),
                op=Operator.ADD,
                right=BinaryOp(
                    left=Variable(name="I"),
                    op=Operator.MUL,
                    right=Variable(name="K", time_index=T_MINUS_1),
                ),
            ),
        )
        model = GCNModel(blocks=[GCNBlock(name="TEST", identities=[eq])])
        names = {v.name for v in model.all_variables()}
        assert names == {"Y", "C", "I", "K"}

    def test_all_parameters_finds_params_in_function_calls(self):
        eq = GCNEquation(
            lhs=Variable(name="Y"),
            rhs=FunctionCall(
                func_name="log",
                args=(
                    BinaryOp(
                        left=Parameter(name="alpha"),
                        op=Operator.MUL,
                        right=Variable(name="K"),
                    ),
                ),
            ),
        )
        model = GCNModel(blocks=[GCNBlock(name="TEST", identities=[eq])])
        params = {p.name for p in model.all_parameters()}
        assert params == {"alpha"}

    def test_all_variables_finds_vars_inside_expectations(self):
        eq = GCNEquation(
            lhs=Variable(name="U"),
            rhs=BinaryOp(
                left=Variable(name="u"),
                op=Operator.ADD,
                right=BinaryOp(
                    left=Parameter(name="beta"),
                    op=Operator.MUL,
                    right=Expectation(expr=Variable(name="U", time_index=T_PLUS_1)),
                ),
            ),
        )
        model = GCNModel(blocks=[GCNBlock(name="TEST", identities=[eq])])
        names = {v.name for v in model.all_variables()}
        assert names == {"U", "u"}
