from gEconpy.parser.ast import (
    T_MINUS_1,
    BinaryOp,
    Expectation,
    FunctionCall,
    GCNEquation,
    NodeTransformer,
    NodeVisitor,
    Number,
    Operator,
    Parameter,
    T,
    TimeIndex,
    UnaryOp,
    Variable,
    collect_nodes_of_type,
)


class TestNodeVisitor:
    def test_dispatches_to_visit_methods(self):
        visited = []

        class TestVisitor(NodeVisitor):
            def visit_Variable(self, node):
                visited.append(node.name)

            def visit_Parameter(self, node):
                visited.append(f"param:{node.name}")

        visitor = TestVisitor()
        expr = BinaryOp(
            left=Variable(name="x", time_index=T),
            op=Operator.ADD,
            right=Parameter(name="alpha"),
        )
        visitor.visit(expr)

        assert "x" in visited
        assert "param:alpha" in visited

    def test_traverses_all_node_types(self):
        names = []

        class NameCollector(NodeVisitor):
            def visit_Variable(self, node):
                names.append(node.name)

        visitor = NameCollector()

        # Nested expression with all node types
        expr = GCNEquation(
            lhs=Variable(name="Y", time_index=T),
            rhs=BinaryOp(
                left=UnaryOp(op=Operator.NEG, operand=Variable(name="a", time_index=T)),
                op=Operator.ADD,
                right=Expectation(
                    expr=FunctionCall(
                        func_name="log",
                        args=(Variable(name="b", time_index=TimeIndex(1)),),
                    )
                ),
            ),
        )
        visitor.visit(expr)

        assert set(names) == {"Y", "a", "b"}


class TestNodeTransformer:
    def test_preserves_unchanged_subtrees(self):
        class SelectiveRenamer(NodeTransformer):
            def visit_Variable(self, node):
                if node.name == "x":
                    return Variable(name="x_new", time_index=node.time_index)
                return node

        transformer = SelectiveRenamer()
        y_node = Variable(name="y", time_index=T)
        expr = BinaryOp(
            left=Variable(name="x", time_index=T),
            op=Operator.ADD,
            right=y_node,
        )
        result = transformer.visit(expr)

        assert result.left.name == "x_new"
        assert result.right is y_node  # Same object - identity preserved

    def test_transforms_all_node_types(self):
        class Renamer(NodeTransformer):
            def visit_Variable(self, node):
                return Variable(name=f"{node.name}_new", time_index=node.time_index)

        transformer = Renamer()

        # Nested expression with all node types
        expr = GCNEquation(
            lhs=Variable(name="Y", time_index=T),
            rhs=BinaryOp(
                left=UnaryOp(op=Operator.NEG, operand=Variable(name="a", time_index=T)),
                op=Operator.ADD,
                right=Expectation(
                    expr=FunctionCall(
                        func_name="log",
                        args=(Variable(name="b", time_index=TimeIndex(1)),),
                    )
                ),
            ),
        )
        result = transformer.visit(expr)

        assert result.lhs.name == "Y_new"
        assert result.rhs.left.operand.name == "a_new"
        assert result.rhs.right.expr.args[0].name == "b_new"

    def test_preserves_equation_metadata(self):
        class Renamer(NodeTransformer):
            def visit_Variable(self, node):
                return Variable(name=f"{node.name}_new", time_index=node.time_index)

        transformer = Renamer()
        eq = GCNEquation(
            lhs=Variable(name="Y", time_index=T),
            rhs=Variable(name="C", time_index=T),
            lagrange_multiplier="lambda",
            calibrating_parameter="beta",
        )
        result = transformer.visit(eq)

        assert result.lagrange_multiplier == "lambda"
        assert result.calibrating_parameter == "beta"


class TestCollectNodesOfType:
    def test_collects_from_nested_expression(self):
        expr = GCNEquation(
            lhs=Variable(name="Y", time_index=T),
            rhs=BinaryOp(
                left=BinaryOp(
                    left=Parameter(name="alpha"),
                    op=Operator.MUL,
                    right=Variable(name="K", time_index=T),
                ),
                op=Operator.ADD,
                right=Expectation(
                    expr=FunctionCall(
                        func_name="log",
                        args=(Variable(name="C", time_index=TimeIndex(1)),),
                    )
                ),
            ),
        )

        variables = collect_nodes_of_type(expr, Variable)
        params = collect_nodes_of_type(expr, Parameter)

        assert {v.name for v in variables} == {"Y", "K", "C"}
        assert {p.name for p in params} == {"alpha"}
