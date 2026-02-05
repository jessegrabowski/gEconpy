import pytest

from gEconpy.parser.ast import (
    STEADY_STATE,
    T_MINUS_1,
    T_PLUS_1,
    BinaryOp,
    Expectation,
    FunctionCall,
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
            Variable(name="C"),  # duplicate
            Variable(name="C", time_index=T_PLUS_1),  # different time
            Variable(name="K"),
        }
        assert len(variables) == 3


class TestExpressionNodes:
    def test_nested_binary_ops_preserve_structure(self):
        # (C + I) * K  vs  C + (I * K) should be different
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
        # log(C / L)
        expr = FunctionCall(
            func_name="log",
            args=(BinaryOp(left=Variable(name="C"), op=Operator.DIV, right=Variable(name="L")),),
        )
        assert "log" in str(expr)
        assert "C" in str(expr)

    def test_expectation_contains_expression(self):
        # E[][beta * U[1]]
        inner = BinaryOp(
            left=Parameter(name="beta"),
            op=Operator.MUL,
            right=Variable(name="U", time_index=T_PLUS_1),
        )
        e = Expectation(expr=inner)
        assert e.expr.left == Parameter(name="beta")


class TestGCNEquation:
    def test_lagrange_and_calibrating_are_exclusive_in_practice(self):
        # Both can technically be set, but semantically one or the other
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


class TestGCNBlock:
    def test_has_optimization_problem_requires_both_controls_and_objective(self):
        # Controls only - no optimization
        block1 = GCNBlock(name="TEST", controls=[Variable(name="C")])
        assert not block1.has_optimization_problem()

        # Objective only - no optimization
        block2 = GCNBlock(
            name="TEST",
            objective=GCNEquation(lhs=Variable(name="U"), rhs=Variable(name="u")),
        )
        assert not block2.has_optimization_problem()

        # Both - has optimization
        block3 = GCNBlock(
            name="TEST",
            controls=[Variable(name="C")],
            objective=GCNEquation(lhs=Variable(name="U"), rhs=Variable(name="u")),
        )
        assert block3.has_optimization_problem()


class TestGCNModel:
    def test_all_variables_traverses_nested_expressions(self):
        # Y = C + I where C and I are in a nested BinaryOp
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
        # log(alpha * K)
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
        # U = u + beta * E[][U[1]]
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
        assert "U" in names
        assert "u" in names
