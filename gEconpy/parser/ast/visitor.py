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

    Subclass and implement ``visit_<NodeType>`` methods for the node types of interest. Node types without a
    dedicated method fall through to :meth:`generic_visit`, which visits the children.

    Examples
    --------
    Collect the name of every variable in an equation:

    .. code-block:: python

        from gEconpy.parser.ast import NodeVisitor
        from gEconpy.parser.grammar.expressions import parse_expression


        class VariableCollector(NodeVisitor):
            def __init__(self):
                self.variables = set()

            def visit_Variable(self, node):
                self.variables.add(node.name)


        collector = VariableCollector()
        collector.visit(parse_expression("A[] * K[-1] ^ alpha"))
        print(collector.variables)
    """

    def visit(self, node: Node) -> Any:
        """
        Dispatch to the ``visit_<NodeType>`` method matching the node's class.

        Parameters
        ----------
        node : Node
            The AST node to visit.

        Returns
        -------
        result : Any
            Whatever the matching visit method returns.
        """
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: Node) -> Any:
        """
        Visit the children of a node. Override to change the default traversal.

        Parameters
        ----------
        node : Node
            The AST node to visit.

        Returns
        -------
        result : Any
            None. Subclasses may return a value.
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
    AST visitor whose visit methods return a node.

    Return the node itself to leave it unchanged, or a new node to replace it. Parent nodes are rebuilt only when a
    child changed, so unchanged subtrees keep their identity.

    Examples
    --------
    Rename one variable throughout an equation:

    .. code-block:: python

        from gEconpy.parser.ast import NodeTransformer, Variable
        from gEconpy.parser.grammar.expressions import parse_expression


        class VariableRenamer(NodeTransformer):
            def visit_Variable(self, node):
                if node.name == "K":
                    return Variable(name="K_new", time_index=node.time_index)
                return node


        renamed = VariableRenamer().visit(parse_expression("A[] * K[-1] ^ alpha"))
        print(renamed)
    """

    def generic_visit(self, node: Node) -> Node:  # noqa: PLR0911
        """
        Transform the children of a node and rebuild the node if any of them changed.

        Parameters
        ----------
        node : Node
            The AST node to transform.

        Returns
        -------
        node : Node
            The original node if no child changed, otherwise a new node with the transformed children.
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

        return node


class NodeCollector[NodeT: Node](NodeVisitor):
    """
    Visitor that collects every node of one type.

    Parameters
    ----------
    node_type : type
        The node class to collect.

    Attributes
    ----------
    collected : set of Node
        The nodes collected so far.
    """

    def __init__(self, node_type: type[NodeT]):
        self.node_type = node_type
        self.collected: set[NodeT] = set()

    def visit(self, node: Node) -> None:
        if isinstance(node, self.node_type):
            self.collected.add(node)
        super().visit(node)


def collect_nodes_of_type[NodeT: Node](node: Node, node_type: type[NodeT]) -> set[NodeT]:
    """
    Collect every node of one type from an AST.

    Parameters
    ----------
    node : Node
        The root node to search.
    node_type : type
        The node class to collect, such as :class:`~gEconpy.parser.ast.Variable`.

    Returns
    -------
    nodes : set of Node
        Every node in the tree that is an instance of ``node_type``.
    """
    collector = NodeCollector(node_type)
    collector.visit(node)
    return collector.collected
