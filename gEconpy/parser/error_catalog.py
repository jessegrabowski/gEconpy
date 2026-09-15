from dataclasses import dataclass, field
from enum import Enum

from gEconpy.parser.constants import (
    BLOCK_COMPONENTS,
    EQUATION_TAGS,
    GCN_ASSUMPTIONS,
    PRELIZ_DIST_WRAPPERS,
)


@dataclass(frozen=True)
class ErrorInfo:
    """
    Catalog entry describing one error code.

    Parameters
    ----------
    title : str
        One-line title of the error.
    explanation : str
        What the error means.
    common_causes : tuple of str, optional
        Situations that produce the error. Defaults to an empty tuple.
    fixes : tuple of str, optional
        Ways to fix the error. Defaults to an empty tuple.
    """

    title: str
    explanation: str
    common_causes: tuple[str, ...] = field(default_factory=tuple)
    fixes: tuple[str, ...] = field(default_factory=tuple)


_BLOCK_COMPONENT_NAMES = ", ".join(name.lower() for name in BLOCK_COMPONENTS)
_EQUATION_TAG_NAMES = ", ".join(f"@{tag}" for tag in EQUATION_TAGS)
_ASSUMPTION_NAMES = ", ".join(GCN_ASSUMPTIONS)
_WRAPPER_NAMES = ", ".join(PRELIZ_DIST_WRAPPERS)


class ErrorCode(Enum):
    """
    Parser error codes, each carrying an :class:`ErrorInfo` catalog entry.

    Codes ``E000`` to ``E099`` are grammar errors raised while parsing. Codes ``E100`` to ``E199`` are semantic
    errors raised during validation. Codes ``W001`` to ``W099`` are warnings.
    """

    E000 = ErrorInfo(
        title="Syntax error",
        explanation="The parser encountered unexpected input.",
        common_causes=(
            "Typo in the source code",
            "Missing or extra punctuation",
            "Invalid expression",
        ),
        fixes=(
            "Read the token the parser stopped on and the statement before it",
            "End every statement with ';' and every block component with '};'",
        ),
    )

    E001 = ErrorInfo(
        title="Missing semicolon",
        explanation="Statements in GCN files must end with a semicolon.",
        common_causes=(
            "Forgot semicolon at end of equation",
            "Forgot semicolon after block component (e.g., 'identities { ... };')",
            "Forgot semicolon after calibration assignment",
        ),
        fixes=(
            "End the statement with ';'",
            "End the block component with '};'",
        ),
    )

    E002 = ErrorInfo(
        title="Unbalanced braces",
        explanation="Opening and closing braces must be balanced. Every '{' needs a matching '}'.",
        common_causes=(
            "Missing closing brace '}'",
            "Extra opening brace '{'",
            "Brace inside a string or comment",
        ),
        fixes=(
            "Close every block and every block component with '};'",
            "Remove the extra '{' or '}'",
        ),
    )

    E003 = ErrorInfo(
        title="Missing block name",
        explanation="Every block must have a name after the 'block' keyword.",
        common_causes=(
            "Forgot to name the block",
            "Typo in 'block' keyword",
        ),
        fixes=("Name the block: block HOUSEHOLD { ... };",),
    )

    E004 = ErrorInfo(
        title="Empty block",
        explanation="A block was declared but contains no components (identities, calibration, etc.).",
        common_causes=(
            "Block not yet implemented",
            "Accidentally deleted block contents",
            "Copy-paste error left empty braces",
        ),
        fixes=(
            "Add at least one component to the block",
            "Delete the block",
        ),
    )

    E005 = ErrorInfo(
        title="Missing equation right-hand side",
        explanation="Equations must have an expression on the right side of the equals sign.",
        common_causes=(
            "Incomplete equation: Y[] = ;",
            "Accidentally deleted the RHS",
            "Copy-paste error",
        ),
        fixes=("Write an expression after '=': Y[] = C[] + I[];",),
    )

    E006 = ErrorInfo(
        title="Missing expression after operator",
        explanation="Binary operators (+, -, *, /, ^) require an expression on both sides.",
        common_causes=(
            "Incomplete expression: Y[] = C[] + ;",
            "Double operator: Y[] = C[] + + I[]",
            "Trailing operator at end of line",
        ),
        fixes=(
            "Write an expression after the operator",
            "Remove the extra operator",
        ),
    )

    E007 = ErrorInfo(
        title="Unbalanced parentheses",
        explanation="Opening and closing parentheses must be balanced.",
        common_causes=(
            "Missing closing parenthesis: log(C[]",
            "Extra opening parenthesis",
            "Nested parentheses not closed properly",
        ),
        fixes=(
            "Close every '(' with ')'",
            "Remove the extra '(' or ')'",
        ),
    )

    E008 = ErrorInfo(
        title="Empty function arguments",
        explanation="Mathematical functions require at least one argument.",
        common_causes=(
            "Empty function call: log()",
            "Missing argument: exp()",
        ),
        fixes=("Pass an argument to the function: log(C[])",),
    )

    E009 = ErrorInfo(
        title="Invalid distribution syntax",
        explanation="Distribution specifications must follow the format: param ~ Distribution(args).",
        common_causes=(
            "Missing tilde: rho Beta(alpha=2, beta=2)",
            "Wrong separator: rho = Beta(alpha=2, beta=2)",
            "Misspelled distribution name",
        ),
        fixes=(
            "Declare a prior with '~': rho ~ Beta(alpha=2, beta=2);",
            "Declare a fixed value with '=': rho = 0.9;",
        ),
    )

    E010 = ErrorInfo(
        title="Invalid variable time index",
        explanation="Time indices in variable brackets must be integers, 'ss' for steady state, or empty.",
        common_causes=(
            "Using letters instead of numbers: Y[abc] instead of Y[]",
            "Typo in 'ss': Y[SS] instead of Y[ss]",
            "Invalid expression in brackets",
        ),
        fixes=("Write the time index as an integer or 'ss': Y[], Y[-1], Y[1], Y[ss]",),
    )

    E011 = ErrorInfo(
        title="Unexpected character",
        explanation="The parser encountered a character that is not valid in GCN syntax.",
        common_causes=(
            "Unicode characters that look like ASCII, such as a curly quote or a minus sign",
            "Special characters pasted from a formatted document",
        ),
        fixes=("Replace the character with its ASCII equivalent",),
    )

    E012 = ErrorInfo(
        title="Missing equals sign",
        explanation="Equations must have an equals sign separating the left-hand side from the right-hand side.",
        common_causes=(
            "Forgot the equals sign: Y[] C[] + I[]",
            "Accidentally deleted the equals sign",
            "Expression written without assignment",
        ),
        fixes=("Separate the two sides with '=': Y[] = C[] + I[];",),
    )

    E013 = ErrorInfo(
        title="Unknown block component",
        explanation=f"Block components must be one of: {_BLOCK_COMPONENT_NAMES}.",
        common_causes=(
            "Typo in component name: 'identites' instead of 'identities'",
            "Using wrong name: 'equations' instead of 'identities'",
            "Made up component name",
        ),
        fixes=(f"Rename the component to one of: {_BLOCK_COMPONENT_NAMES}",),
    )

    E014 = ErrorInfo(
        title="Unknown equation tag",
        explanation=f"Equation tags must be one of: {_EQUATION_TAG_NAMES}.",
        common_causes=("Typo in tag name",),
        fixes=(f"Rename the tag to one of: {_EQUATION_TAG_NAMES}",),
    )

    E015 = ErrorInfo(
        title="Unknown assumption",
        explanation=f"Assumption names must be one of: {_ASSUMPTION_NAMES}.",
        common_causes=("Typo in assumption name: 'postive' for 'positive'",),
        fixes=(f"Rename the assumption to one of: {_ASSUMPTION_NAMES}",),
    )

    E016 = ErrorInfo(
        title="Component outside block",
        explanation="Block components like 'identities', 'calibration', etc. must appear inside a block.",
        common_causes=(
            "Missing 'block NAME {' before component",
            "Closed the block too early",
            "Typo in 'block' keyword",
        ),
        fixes=(
            "Wrap the component in a block: block NAME { ... };",
            "Move the '};' that closes the block below the component",
        ),
    )

    E100 = ErrorInfo(
        title="Duplicate block name",
        explanation="Each block must have a unique name within the model.",
        common_causes=(
            "Copy-pasted a block and forgot to rename it",
            "Two blocks accidentally given the same name",
        ),
        fixes=(
            "Rename one of the blocks",
            "Merge the two blocks into one",
        ),
    )

    E101 = ErrorInfo(
        title="Duplicate parameter definition",
        explanation="A parameter can only be calibrated once across all blocks.",
        common_causes=(
            "Same parameter calibrated in multiple blocks",
            "Parameter accidentally defined twice in same calibration section",
        ),
        fixes=(
            "Delete one of the definitions",
            "Rename one parameter if the two are distinct",
        ),
    )

    E102 = ErrorInfo(
        title="Unknown distribution",
        explanation="The distribution name is not recognized.",
        common_causes=(
            "Typo in distribution name: 'Betta' for 'Beta'",
            "A distribution PreliZ does not define",
        ),
        fixes=("Use a PreliZ distribution name with its capitalization: Normal, Beta, Gamma, InverseGamma, Uniform",),
    )

    E103 = ErrorInfo(
        title="Unknown wrapper",
        explanation=f"The distribution wrapper must be one of: {_WRAPPER_NAMES}.",
        common_causes=("Typo in wrapper name: 'Truncatd' for 'Truncated'",),
        fixes=(f"Rename the wrapper to one of: {_WRAPPER_NAMES}",),
    )

    W001 = ErrorInfo(
        title="Unused parameter",
        explanation="A parameter is calibrated but never used in any equation.",
        common_causes=(
            "Parameter defined for future use",
            "Equation using the parameter was deleted",
            "Typo in parameter name in equations",
        ),
        fixes=(
            "Delete the parameter from the calibration block",
            "Fix the spelling of the parameter in the equation that should use it",
        ),
    )

    W002 = ErrorInfo(
        title="Controls without objective",
        explanation="Control variables are defined but no objective function is specified.",
        common_causes=(
            "Objective function not yet added",
            "Block is meant to be an identity block, not optimization",
        ),
        fixes=(
            "Add an objective component",
            "Delete the controls component",
        ),
    )

    W003 = ErrorInfo(
        title="Objective without controls",
        explanation="An objective function is specified but no control variables are defined.",
        common_causes=(
            "Forgot to add controls section",
            "Controls defined in wrong block",
        ),
        fixes=(
            "Add a controls component listing the choice variables",
            "Move the objective to the block that declares the controls",
        ),
    )

    @property
    def info(self) -> ErrorInfo:
        """The catalog entry for this code: title, explanation, common causes, and fixes."""
        return self.value

    @property
    def title(self) -> str:
        """The one-line title of this code."""
        return self.value.title

    @property
    def explanation(self) -> str:
        """The prose explanation of this code."""
        return self.value.explanation

    def format_help(self) -> str:
        """
        Render the full catalog entry for this code as help text.

        Returns
        -------
        help_text : str
            The title, explanation, common causes, and fixes, ready to print.
        """
        info = self.value
        lines = [
            f"{self.name}: {info.title}",
            "",
            info.explanation,
            "",
            "Common causes:",
        ]
        lines.extend(f"  - {cause}" for cause in info.common_causes)

        lines.append("")
        lines.append("How to fix:")
        lines.extend(f"  - {fix}" for fix in info.fixes)

        return "\n".join(lines)


def get_error_info(code: str | ErrorCode) -> ErrorInfo | None:
    """
    Look up the catalog entry for an error code.

    Parameters
    ----------
    code : str or ErrorCode
        The error code, as a member of :class:`ErrorCode` or its name such as ``"E001"``.

    Returns
    -------
    info : ErrorInfo or None
        The catalog entry, or None when no code has that name.
    """
    error_code = _lookup_code(code)
    return error_code.value if error_code is not None else None


def format_error_help(code: str | ErrorCode) -> str:
    """
    Render the full catalog entry for an error code as help text.

    Parameters
    ----------
    code : str or ErrorCode
        The error code, as a member of :class:`ErrorCode` or its name such as ``"E001"``.

    Returns
    -------
    help_text : str
        The title, explanation, common causes, and fixes, or an empty string when no code has that name.
    """
    error_code = _lookup_code(code)
    return error_code.format_help() if error_code is not None else ""


def _lookup_code(code: str | ErrorCode) -> ErrorCode | None:
    if isinstance(code, ErrorCode):
        return code
    try:
        return ErrorCode[code]
    except KeyError:
        return None
