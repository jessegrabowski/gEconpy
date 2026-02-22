from typing import Any

from gEconpy.parser.ast.nodes import (
    BinaryOp,
    Expectation,
    FunctionCall,
    GCNEquation,
    Node,
    UnaryOp,
)


class NodeVisitor:
    """
    Base class for AST visitors.

    Subclass and implement visit_<NodeType> methods to handle specific node types.
    Unhandled node types fall through to `generic_visit`, which recursively visits
    children.

    Examples
    --------
    .. code-block:: python

        class VariableCollector(NodeVisitor):
            def __init__(self):
                self.variables = set()

            def visit_Variable(self, node):
                self.variables.add(node.name)


        collector = VariableCollector()
        collector.visit(some_equation)
        print(collector.variables)
    """

    def visit(self, node: Node) -> Any:
        """
        Dispatch to the appropriate visit method based on node type.

        Parameters
        ----------
        node : Node
            The AST node to visit.

        Returns
        -------
        Any
            Result of the visit method.
        """
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: Node) -> None:
        """
        Default visitor that recursively visits child nodes.

        Override this to change default traversal behavior.

        Parameters
        ----------
        node : Node
            The AST node to visit.
        """
        if isinstance(node, BinaryOp):
            self.visit(node.left)
            self.visit(node.right)
        elif isinstance(node, UnaryOp):
            self.visit(node.operand)
        elif isinstance(node, FunctionCall):
            for arg in node.args:
                self.visit(arg)
        elif isinstance(node, Expectation):
            self.visit(node.expr)
        elif isinstance(node, GCNEquation):
            self.visit(node.lhs)
            self.visit(node.rhs)


class NodeTransformer(NodeVisitor):
    """
    AST visitor that can transform nodes.

    Similar to `NodeVisitor`, but visit methods should return a node.
    Return the same node to keep it unchanged, or a new node to replace it.
    The transformer automatically handles structural changes, only creating
    new parent nodes when children have changed.

    Examples
    --------
    .. code-block:: python

        class VariableRenamer(NodeTransformer):
            def visit_Variable(self, node):
                if node.name == "old_name":
                    return Variable(name="new_name", time_index=node.time_index)
                return node


        transformer = VariableRenamer()
        new_equation = transformer.visit(old_equation)
    """

    def generic_visit(self, node: Node) -> Node:  # noqa: PLR0911
        """
        Default transformer that recursively transforms child nodes.

        Creates new parent nodes only when children have actually changed,
        preserving object identity for unchanged subtrees.

        Parameters
        ----------
        node : Node
            The AST node to transform.

        Returns
        -------
        Node
            The original node if unchanged, or a new node with transformed children.
        """
        if isinstance(node, BinaryOp):
            new_left = self.visit(node.left)
            new_right = self.visit(node.right)
            if new_left is node.left and new_right is node.right:
                return node
            return BinaryOp(left=new_left, op=node.op, right=new_right, location=node.location)

        if isinstance(node, UnaryOp):
            new_operand = self.visit(node.operand)
            if new_operand is node.operand:
                return node
            return UnaryOp(op=node.op, operand=new_operand, location=node.location)

        if isinstance(node, FunctionCall):
            new_args = tuple(self.visit(arg) for arg in node.args)
            if all(new is orig for new, orig in zip(new_args, node.args, strict=True)):
                return node
            return FunctionCall(func_name=node.func_name, args=new_args, location=node.location)

        if isinstance(node, Expectation):
            new_expr = self.visit(node.expr)
            if new_expr is node.expr:
                return node
            return Expectation(expr=new_expr, location=node.location)

        if isinstance(node, GCNEquation):
            new_lhs = self.visit(node.lhs)
            new_rhs = self.visit(node.rhs)
            if new_lhs is node.lhs and new_rhs is node.rhs:
                return node
            return GCNEquation(
                lhs=new_lhs,
                rhs=new_rhs,
                lagrange_multiplier=node.lagrange_multiplier,
                calibrating_parameter=node.calibrating_parameter,
                tags=node.tags,
                location=node.location,
            )

        # Leaf nodes (Number, Parameter, Variable) - return as-is
        return node


class NodeCollector(NodeVisitor):
    """
    Visitor that collects all nodes of a specific type.

    Parameters
    ----------
    node_type : type
        The type of node to collect.

    Attributes
    ----------
    collected : set
        Set of all collected nodes.
    """

    def __init__(self, node_type: type):
        self.node_type = node_type
        self.collected: set = set()

    def visit(self, node: Node) -> None:
        if isinstance(node, self.node_type):
            self.collected.add(node)
        super().visit(node)


def collect_nodes_of_type(node: Node, node_type: type) -> set:
    """
    Recursively collect all nodes of a specific type from an AST.

    Parameters
    ----------
    node : Node
        The root node to search.
    node_type : type
        The type of node to collect (e.g., Variable, Parameter).

    Returns
    -------
    set
        Set of all nodes of the specified type.
    """
    collector = NodeCollector(node_type)
    collector.visit(node)
    return collector.collected
