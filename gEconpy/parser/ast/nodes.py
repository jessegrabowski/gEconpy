from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Self, cast

from gEconpy.parser.errors import ParseLocation


class TimeIndex:
    """
    The time index of a variable: an integer offset from the current period, or the steady state.

    Parameters
    ----------
    value : int or str
        An integer offset (0 for ``t``, 1 for ``t+1``, -1 for ``t-1``), or the string ``"ss"`` for the steady state.
    """

    __slots__ = ("_value",)

    STEADY_STATE = "ss"

    def __init__(self, value: int | str):
        if isinstance(value, str) and value != self.STEADY_STATE:
            raise ValueError(f"String time index must be 'ss', got '{value}'")
        self._value = value

    @property
    def is_steady_state(self) -> bool:
        return self._value == self.STEADY_STATE

    @property
    def value(self) -> int | str:
        return self._value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TimeIndex):
            return self._value == other._value
        return self._value == other

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        if self.is_steady_state:
            return "TimeIndex(ss)"
        return f"TimeIndex({self._value})"

    def __str__(self) -> str:
        if self.is_steady_state:
            return "[ss]"
        if self._value == 0:
            return "[]"
        return f"[{self._value}]"

    def step_forward(self) -> Self:
        if self.is_steady_state:
            raise ValueError("Cannot step forward from steady state")
        return type(self)(cast(int, self._value) + 1)

    def step_backward(self) -> Self:
        if self.is_steady_state:
            raise ValueError("Cannot step backward from steady state")
        return type(self)(cast(int, self._value) - 1)


T = TimeIndex(0)
T_PLUS_1 = TimeIndex(1)
T_MINUS_1 = TimeIndex(-1)
STEADY_STATE = TimeIndex("ss")


class Operator(Enum):
    """Binary and unary operators in expressions. ``NEG`` is the only unary operator."""

    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    NEG = auto()

    def __str__(self) -> str:
        symbols = {
            Operator.ADD: "+",
            Operator.SUB: "-",
            Operator.MUL: "*",
            Operator.DIV: "/",
            Operator.POW: "^",
            Operator.NEG: "-",
        }
        return symbols[self]


class BlockComponent(Enum):
    """Valid component types within a GCN block."""

    DEFINITIONS = "definitions"
    CONTROLS = "controls"
    OBJECTIVE = "objective"
    CONSTRAINTS = "constraints"
    IDENTITIES = "identities"
    SHOCKS = "shocks"
    CALIBRATION = "calibration"


class Tag(Enum):
    """
    Tags applied to GCN equations with the ``@tag`` syntax.

    ``EXCLUDE`` drops the equation from the final system after optimization. ``MINIMIZE`` makes the objective a
    minimization problem, and ``MAXIMIZE`` a maximization problem, which is the default.
    """

    EXCLUDE = "exclude"
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

    @classmethod
    def from_string(cls, name: str) -> Self:
        """
        Look up a tag by name, ignoring case.

        Parameters
        ----------
        name : str
            Tag name, written without the leading ``@``.

        Returns
        -------
        tag : Tag
            The matching tag.
        """
        name_lower = name.lower()
        for tag in cls:
            if tag.value == name_lower:
                return tag
        valid_tags = ", ".join(t.value for t in cls)
        raise ValueError(f"Unknown tag '@{name}'. Valid tags are: {valid_tags}")

    def __str__(self) -> str:
        return f"@{self.value}"


@dataclass(frozen=True)
class Node:
    """Base class for all AST nodes."""

    location: ParseLocation | None = field(default=None, compare=False, repr=False, kw_only=True)

    def with_location(self, location: ParseLocation) -> Self:
        raise NotImplementedError


@dataclass(frozen=True)
class Number(Node):
    """A numeric literal."""

    value: float

    def with_location(self, location: ParseLocation) -> Self:
        return type(self)(value=self.value, location=location)

    def __str__(self) -> str:
        if self.value == int(self.value):
            return str(int(self.value))
        return str(self.value)


@dataclass(frozen=True)
class Parameter(Node):
    """A model parameter such as ``alpha`` or ``beta``. Parameters carry no time index."""

    name: str

    def with_location(self, location: ParseLocation) -> Self:
        return type(self)(name=self.name, location=location)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Variable(Node):
    """A time-indexed model variable such as ``C[]``, ``K[-1]``, ``Y[1]``, or ``A[ss]``."""

    name: str
    time_index: TimeIndex = field(default_factory=lambda: T)

    def with_location(self, location: ParseLocation) -> Self:
        return type(self)(name=self.name, time_index=self.time_index, location=location)

    def at(self, time_index: TimeIndex | int | str) -> Self:
        """
        Return a copy of this variable at a different time index.

        Parameters
        ----------
        time_index : TimeIndex or int or str
            The new time index. An integer or string is converted to a TimeIndex.

        Returns
        -------
        variable : Variable
            The variable at the new time index.
        """
        if isinstance(time_index, int | str):
            time_index = TimeIndex(time_index)
        return type(self)(name=self.name, time_index=time_index, location=self.location)

    def to_ss(self) -> Self:
        """
        Return a copy of this variable at the steady state.

        Returns
        -------
        variable : Variable
            The steady-state variable.
        """
        return self.at(STEADY_STATE)

    def __str__(self) -> str:
        return f"{self.name}{self.time_index}"


@dataclass(frozen=True)
class BinaryOp(Node):
    """A binary operation ``left op right``."""

    left: Node
    op: Operator
    right: Node

    def with_location(self, location: ParseLocation) -> Self:
        return type(self)(left=self.left, op=self.op, right=self.right, location=location)

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass(frozen=True)
class UnaryOp(Node):
    """A unary operation ``op operand``."""

    op: Operator
    operand: Node

    def with_location(self, location: ParseLocation) -> Self:
        return type(self)(op=self.op, operand=self.operand, location=location)

    def __str__(self) -> str:
        return f"({self.op}{self.operand})"


@dataclass(frozen=True)
class FunctionCall(Node):
    """A function call such as ``log(x)`` or ``exp(y)``. The function is stored by name."""

    func_name: str
    args: tuple[Node, ...]

    def with_location(self, location: ParseLocation) -> Self:
        return type(self)(func_name=self.func_name, args=self.args, location=location)

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.func_name}({args_str})"


@dataclass(frozen=True)
class Expectation(Node):
    """The expectation operator ``E[][...]`` wrapped around an expression."""

    expr: Node

    def with_location(self, location: ParseLocation) -> Self:
        return type(self)(expr=self.expr, location=location)

    def __str__(self) -> str:
        return f"E[][{self.expr}]"


@dataclass(frozen=True)
class GCNEquation(Node):
    """
    A model equation ``lhs = rhs``.

    A constraint may name a Lagrange multiplier, a calibration equation may name the parameter it calibrates with
    ``-> param``, and any equation may carry tags such as ``@exclude``.
    """

    lhs: Node
    rhs: Node
    lagrange_multiplier: str | None = None
    calibrating_parameter: str | None = None
    tags: frozenset[Tag] = field(default_factory=frozenset)

    def with_location(self, location: ParseLocation) -> Self:
        return type(self)(
            lhs=self.lhs,
            rhs=self.rhs,
            lagrange_multiplier=self.lagrange_multiplier,
            calibrating_parameter=self.calibrating_parameter,
            tags=self.tags,
            location=location,
        )

    def with_tags(self, tags: frozenset[Tag]) -> Self:
        """
        Return a copy of this equation carrying a different set of tags.

        Parameters
        ----------
        tags : frozenset of Tag
            Tags for the new equation. They replace the current tags.

        Returns
        -------
        equation : GCNEquation
            The tagged equation.
        """
        return type(self)(
            lhs=self.lhs,
            rhs=self.rhs,
            lagrange_multiplier=self.lagrange_multiplier,
            calibrating_parameter=self.calibrating_parameter,
            tags=tags,
            location=self.location,
        )

    def has_tag(self, tag: Tag) -> bool:
        return tag in self.tags

    @property
    def is_excluded(self) -> bool:
        return Tag.EXCLUDE in self.tags

    @property
    def is_minimize(self) -> bool:
        return Tag.MINIMIZE in self.tags

    @property
    def is_maximize(self) -> bool:
        return Tag.MAXIMIZE in self.tags

    @property
    def is_calibrating(self) -> bool:
        return self.calibrating_parameter is not None

    @property
    def has_lagrange_multiplier(self) -> bool:
        return self.lagrange_multiplier is not None

    def __str__(self) -> str:
        tag_lines = [str(tag) for tag in sorted(self.tags, key=lambda t: t.value)]
        equation = "\n".join([*tag_lines, f"{self.lhs} = {self.rhs}"])
        if self.lagrange_multiplier:
            equation += f" : {self.lagrange_multiplier}"
        if self.calibrating_parameter:
            equation += f" -> {self.calibrating_parameter}"
        return equation


@dataclass(frozen=True)
class GCNDistribution(Node):
    """
    A prior distribution declaration such as ``alpha ~ Beta(mean=0.5, sd=0.1) = 0.35``.

    The distribution may be wrapped, as in ``beta ~ maxent(Normal(), lower=0, upper=1) = 0.5``.
    """

    parameter_name: str
    dist_name: str
    dist_kwargs: dict[str, float | str | None] = field(default_factory=dict)
    wrapper_name: str | None = None
    wrapper_kwargs: dict[str, float | str | None] = field(default_factory=dict)
    initial_value: float | None = None

    def with_location(self, location: ParseLocation) -> Self:
        return type(self)(
            parameter_name=self.parameter_name,
            dist_name=self.dist_name,
            dist_kwargs=self.dist_kwargs,
            wrapper_name=self.wrapper_name,
            wrapper_kwargs=self.wrapper_kwargs,
            initial_value=self.initial_value,
            location=location,
        )

    @property
    def is_wrapped(self) -> bool:
        return self.wrapper_name is not None

    def __str__(self) -> str:
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.dist_kwargs.items())
        dist_str = f"{self.dist_name}({kwargs_str})"

        if self.wrapper_name:
            wrapper_args = [dist_str, *(f"{k}={v}" for k, v in self.wrapper_kwargs.items())]
            dist_str = f"{self.wrapper_name}({', '.join(wrapper_args)})"

        declaration = f"{self.parameter_name} ~ {dist_str}"
        if self.initial_value is not None:
            declaration += f" = {self.initial_value}"
        return declaration


@dataclass
class GCNBlock:
    """
    A named model block, such as ``HOUSEHOLD`` or ``FIRM``, holding its equations by component.

    Parameters
    ----------
    name : str
        The block name.
    definitions : list of GCNEquation, optional
        Equations from the ``definitions`` component. Defaults to an empty list.
    controls : list of Variable, optional
        Variables from the ``controls`` component. Defaults to an empty list.
    objective : list of GCNEquation, optional
        Equations from the ``objective`` component. Defaults to an empty list.
    constraints : list of GCNEquation, optional
        Equations from the ``constraints`` component. Defaults to an empty list.
    identities : list of GCNEquation, optional
        Equations from the ``identities`` component. Defaults to an empty list.
    shocks : list of Variable, optional
        Variables from the ``shocks`` component. Defaults to an empty list.
    shock_distributions : list of GCNDistribution, optional
        Prior distributions declared on shocks. Defaults to an empty list.
    calibration : list of GCNEquation or GCNDistribution, optional
        Entries from the ``calibration`` component. Defaults to an empty list.
    location : ParseLocation, optional
        Where the block starts in the source. Defaults to None.
    """

    name: str
    definitions: list[GCNEquation] = field(default_factory=list)
    controls: list[Variable] = field(default_factory=list)
    objective: list[GCNEquation] = field(default_factory=list)
    constraints: list[GCNEquation] = field(default_factory=list)
    identities: list[GCNEquation] = field(default_factory=list)
    shocks: list[Variable] = field(default_factory=list)
    shock_distributions: list[GCNDistribution] = field(default_factory=list)
    calibration: list[GCNEquation | GCNDistribution] = field(default_factory=list)
    location: ParseLocation | None = field(default=None, compare=False, repr=False)

    def get_component(self, component: BlockComponent) -> list[GCNEquation | GCNDistribution | Variable]:
        return getattr(self, component.value)

    def has_optimization_problem(self) -> bool:
        return len(self.controls) > 0 and len(self.objective) > 0


@dataclass
class GCNModel:
    """
    The root of a parsed GCN file: its blocks and the ``options``, ``tryreduce``, and ``assumptions`` sections.

    Parameters
    ----------
    blocks : list of GCNBlock, optional
        The model blocks in source order. Defaults to an empty list.
    options : dict mapping str to str or bool, optional
        Entries of the ``options`` section. Defaults to an empty dict.
    tryreduce : list of str, optional
        Variable names listed in the ``tryreduce`` section. Defaults to an empty list.
    assumptions : dict mapping str to dict, optional
        SymPy assumptions per symbol name, as in ``{"C": {"positive": True}}``. Defaults to an empty dict.
    filename : str, optional
        Path of the source file. Defaults to an empty string.
    """

    blocks: list[GCNBlock] = field(default_factory=list)
    options: dict[str, str | bool] = field(default_factory=dict)
    tryreduce: list[str] = field(default_factory=list)
    assumptions: dict[str, dict[str, bool]] = field(default_factory=dict)
    filename: str = ""

    def get_block(self, name: str) -> GCNBlock | None:
        for block in self.blocks:
            if block.name == name:
                return block
        return None

    def block_names(self) -> list[str]:
        return [block.name for block in self.blocks]

    def all_equations(self) -> list[GCNEquation]:
        """
        Collect the definitions, objectives, constraints, and identities of every block.

        Returns
        -------
        equations : list of GCNEquation
            The collected equations, in block order.
        """
        equations = []
        for block in self.blocks:
            equations.extend(block.definitions)
            equations.extend(block.objective)
            equations.extend(block.constraints)
            equations.extend(block.identities)
        return equations

    def all_variables(self) -> set[Variable]:
        """
        Collect the variables appearing in the equations of every block.

        Returns
        -------
        variables : set of Variable
            The collected variables. Two time indices of one name count as two variables.
        """
        variables: set[Variable] = set()
        for eq in self.all_equations():
            variables.update(collect_nodes_of_type(eq.lhs, Variable))
            variables.update(collect_nodes_of_type(eq.rhs, Variable))
        return variables

    def all_parameters(self) -> set[Parameter]:
        """
        Collect the parameters appearing in the equations of every block.

        Returns
        -------
        parameters : set of Parameter
            The collected parameters.
        """
        parameters: set[Parameter] = set()
        for eq in self.all_equations():
            parameters.update(collect_nodes_of_type(eq.lhs, Parameter))
            parameters.update(collect_nodes_of_type(eq.rhs, Parameter))
        return parameters


def collect_nodes_of_type[NodeT: Node](node: Node, node_type: type[NodeT]) -> set[NodeT]:
    """
    Collect every node of one type from an expression tree.

    Parameters
    ----------
    node : Node
        Root of the expression to search.
    node_type : type
        The node class to collect, such as :class:`Variable` or :class:`Parameter`.

    Returns
    -------
    nodes : set of Node
        Every node in the tree that is an instance of ``node_type``.
    """
    # Deferred import: the visitor module imports the node classes from this module.
    from gEconpy.parser.ast.visitor import collect_nodes_of_type as visitor_collect  # noqa: PLC0415

    return visitor_collect(node, node_type)


def collect_variable_names(node: Node) -> set[str]:
    """
    Collect the names of every variable in an expression, ignoring time indices.

    Parameters
    ----------
    node : Node
        Root of the expression to search.

    Returns
    -------
    names : set of str
        The variable names found.
    """
    return {v.name for v in collect_nodes_of_type(node, Variable)}


def collect_parameter_names(node: Node) -> set[str]:
    """
    Collect the names of every parameter in an expression.

    Parameters
    ----------
    node : Node
        Root of the expression to search.

    Returns
    -------
    names : set of str
        The parameter names found.
    """
    return {p.name for p in collect_nodes_of_type(node, Parameter)}
