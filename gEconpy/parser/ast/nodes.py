from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from gEconpy.parser.errors import ParseLocation


class TimeIndex:
    """
    Represents a time index for a variable.

    Can be an integer offset (0 for t, 1 for t+1, -1 for t-1) or steady-state.
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
        if self._value > 0:
            return f"[{self._value}]"
        return f"[{self._value}]"

    def step_forward(self) -> TimeIndex:
        if self.is_steady_state:
            raise ValueError("Cannot step forward from steady state")
        return TimeIndex(self._value + 1)

    def step_backward(self) -> TimeIndex:
        if self.is_steady_state:
            raise ValueError("Cannot step backward from steady state")
        return TimeIndex(self._value - 1)


# Singleton instances for common time indices
T = TimeIndex(0)
T_PLUS_1 = TimeIndex(1)
T_MINUS_1 = TimeIndex(-1)
STEADY_STATE = TimeIndex("ss")


class Operator(Enum):
    """Binary and unary operators in expressions."""

    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    NEG = auto()  # Unary negation

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
    Tags that can be applied to GCN objects using @tag syntax.

    Tags modify how objects are processed during model building.
    """

    EXCLUDE = "exclude"  # Exclude equation from final system after optimization

    @classmethod
    def from_string(cls, name: str) -> Tag:
        """Convert a string to a Tag, raising ValueError if unknown."""
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

    def with_location(self, location: ParseLocation) -> Node:
        raise NotImplementedError


# --- Expression Nodes ---


@dataclass(frozen=True)
class Number(Node):
    """A numeric literal."""

    value: float

    def with_location(self, location: ParseLocation) -> Number:
        return Number(value=self.value, location=location)

    def __str__(self) -> str:
        if self.value == int(self.value):
            return str(int(self.value))
        return str(self.value)


@dataclass(frozen=True)
class Parameter(Node):
    """
    A model parameter (no time index).

    Parameters are constants that don't vary over time, like `alpha`, `beta`, `delta`.
    """

    name: str

    def with_location(self, location: ParseLocation) -> Parameter:
        return Parameter(name=self.name, location=location)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Variable(Node):
    """
    A time-indexed model variable.

    Variables have a base name and a time index, like `C[]`, `K[-1]`, `Y[1]`, `A[ss]`.
    """

    name: str
    time_index: TimeIndex = field(default_factory=lambda: T)

    def with_location(self, location: ParseLocation) -> Variable:
        return Variable(name=self.name, time_index=self.time_index, location=location)

    def at(self, time_index: TimeIndex | int | str) -> Variable:
        """Return a new Variable at a different time index."""
        if isinstance(time_index, int | str):
            time_index = TimeIndex(time_index)
        return Variable(name=self.name, time_index=time_index, location=self.location)

    def to_ss(self) -> Variable:
        """Return steady-state version of this variable."""
        return self.at(STEADY_STATE)

    def __str__(self) -> str:
        return f"{self.name}{self.time_index}"


@dataclass(frozen=True)
class BinaryOp(Node):
    """A binary operation: left op right."""

    left: Node
    op: Operator
    right: Node

    def with_location(self, location: ParseLocation) -> BinaryOp:
        return BinaryOp(left=self.left, op=self.op, right=self.right, location=location)

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass(frozen=True)
class UnaryOp(Node):
    """A unary operation: op operand."""

    op: Operator
    operand: Node

    def with_location(self, location: ParseLocation) -> UnaryOp:
        return UnaryOp(op=self.op, operand=self.operand, location=location)

    def __str__(self) -> str:
        return f"({self.op}{self.operand})"


@dataclass(frozen=True)
class FunctionCall(Node):
    """
    A function call like `log(x)` or `exp(y)`.

    The function name is stored as a string, and arguments are expression nodes.
    """

    func_name: str
    args: tuple[Node, ...]

    def with_location(self, location: ParseLocation) -> FunctionCall:
        return FunctionCall(func_name=self.func_name, args=self.args, location=location)

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.func_name}({args_str})"


@dataclass(frozen=True)
class Expectation(Node):
    """
    The expectation operator E[][...].

    Contains the expression inside the expectation.
    """

    expr: Node

    def with_location(self, location: ParseLocation) -> Expectation:
        return Expectation(expr=self.expr, location=location)

    def __str__(self) -> str:
        return f"E[][{self.expr}]"


# --- Equation Node ---


@dataclass(frozen=True)
class GCNEquation(Node):
    """
    A model equation: lhs = rhs.

    Equations can optionally have:
    - A Lagrange multiplier name (for constraints)
    - A calibrating parameter (for calibration equations with ->)
    - Tags that modify processing (e.g., @exclude)
    """

    lhs: Node
    rhs: Node
    lagrange_multiplier: str | None = None
    calibrating_parameter: str | None = None
    tags: frozenset[Tag] = field(default_factory=frozenset)

    def with_location(self, location: ParseLocation) -> GCNEquation:
        return GCNEquation(
            lhs=self.lhs,
            rhs=self.rhs,
            lagrange_multiplier=self.lagrange_multiplier,
            calibrating_parameter=self.calibrating_parameter,
            tags=self.tags,
            location=location,
        )

    def with_tags(self, tags: frozenset[Tag]) -> GCNEquation:
        """Return a new equation with the specified tags."""
        return GCNEquation(
            lhs=self.lhs,
            rhs=self.rhs,
            lagrange_multiplier=self.lagrange_multiplier,
            calibrating_parameter=self.calibrating_parameter,
            tags=tags,
            location=self.location,
        )

    def has_tag(self, tag: Tag) -> bool:
        """Check if the equation has a specific tag."""
        return tag in self.tags

    @property
    def is_excluded(self) -> bool:
        """Check if the equation has the @exclude tag."""
        return Tag.EXCLUDE in self.tags

    @property
    def is_calibrating(self) -> bool:
        return self.calibrating_parameter is not None

    @property
    def has_lagrange_multiplier(self) -> bool:
        return self.lagrange_multiplier is not None

    def __str__(self) -> str:
        parts = []
        if self.tags:
            parts.extend(str(tag) for tag in sorted(self.tags, key=lambda t: t.value))
        parts.append(f"{self.lhs} = {self.rhs}")
        base = "\n".join(parts) if self.tags else parts[0]
        if self.lagrange_multiplier:
            base += f" : {self.lagrange_multiplier}"
        if self.calibrating_parameter:
            base = f"{base} -> {self.calibrating_parameter}"
        return base


# --- Distribution Node ---


@dataclass(frozen=True)
class GCNDistribution(Node):
    """
    A prior distribution declaration.

    Represents things like `alpha ~ Beta(mean=0.5, sd=0.1) = 0.35`
    or wrapped distributions like `maxent(Normal(), lower=0, upper=1) = 0.5`.
    """

    parameter_name: str
    dist_name: str
    dist_kwargs: dict[str, float | str] = field(default_factory=dict)
    wrapper_name: str | None = None
    wrapper_kwargs: dict[str, float | None] = field(default_factory=dict)
    initial_value: float | None = None

    def with_location(self, location: ParseLocation) -> GCNDistribution:
        return GCNDistribution(
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
            wrapper_kwargs_str = ", ".join(f"{k}={v}" for k, v in self.wrapper_kwargs.items())
            if wrapper_kwargs_str:
                dist_str = f"{self.wrapper_name}({dist_str}, {wrapper_kwargs_str})"
            else:
                dist_str = f"{self.wrapper_name}({dist_str})"

        result = f"{self.parameter_name} ~ {dist_str}"
        if self.initial_value is not None:
            result += f" = {self.initial_value}"
        return result


# --- Block Node ---


@dataclass
class GCNBlock:
    """
    A model block containing equations organized by component type.

    Blocks have a name (like "HOUSEHOLD", "FIRM") and contain equations
    organized into components (definitions, controls, objective, etc.).
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

    def get_component(self, component: BlockComponent) -> list | GCNEquation | None:
        return getattr(self, component.value)

    def has_optimization_problem(self) -> bool:
        return len(self.controls) > 0 and len(self.objective) > 0


# --- Model Node (Root) ---


@dataclass
class GCNModel:
    """
    The root AST node representing a complete GCN model.

    Contains all blocks, special block data (options, tryreduce, assumptions),
    and tracks any prior distributions defined.
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
        """Collect all equations from all blocks."""
        equations = []
        for block in self.blocks:
            equations.extend(block.definitions)
            equations.extend(block.objective)
            equations.extend(block.constraints)
            equations.extend(block.identities)
        return equations

    def all_variables(self) -> set[Variable]:
        """Collect all unique variables from all blocks."""
        variables: set[Variable] = set()
        for eq in self.all_equations():
            variables.update(_collect_variables(eq.lhs))
            variables.update(_collect_variables(eq.rhs))
        return variables

    def all_parameters(self) -> set[Parameter]:
        """Collect all unique parameters from all blocks."""
        parameters: set[Parameter] = set()
        for eq in self.all_equations():
            parameters.update(_collect_parameters(eq.lhs))
            parameters.update(_collect_parameters(eq.rhs))
        return parameters


def _collect_variables(node: Node) -> set[Variable]:
    """Recursively collect all Variable nodes from an expression."""
    return collect_nodes_of_type(node, Variable)


def _collect_parameters(node: Node) -> set[Parameter]:
    """Recursively collect all Parameter nodes from an expression."""
    return collect_nodes_of_type(node, Parameter)


def collect_nodes_of_type(node: Node, node_type: type) -> set:
    """
    Recursively collect all nodes of a specific type from an expression.

    Parameters
    ----------
    node : Node
        The expression node to search.
    node_type : type
        The type of node to collect (e.g., Variable, Parameter).

    Returns
    -------
    result : set
        Set of all nodes of the specified type.
    """
    # Deferred import to avoid circular dependency (visitor imports from nodes)
    from gEconpy.parser.ast.visitor import collect_nodes_of_type as _collect  # noqa: PLC0415

    return _collect(node, node_type)


def collect_variable_names(node: Node) -> set[str]:
    """Recursively collect all variable names from an expression."""
    return {v.name for v in collect_nodes_of_type(node, Variable)}


def collect_parameter_names(node: Node) -> set[str]:
    """Recursively collect all parameter names from an expression."""
    return {p.name for p in collect_nodes_of_type(node, Parameter)}
