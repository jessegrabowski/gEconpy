from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.solvers.gensys import interpret_gensys_output

if TYPE_CHECKING:
    from gEconpy.parser.errors import ParseLocation


class GCNValidationError(ValueError):
    """
    Base class for model validation errors that display the offending source location.

    Parameters
    ----------
    message : str
        The main error message.
    source : str, optional
        The full source code of the GCN file.
    location : ParseLocation, optional
        The location of the offending token in the source. Source context is shown only when both ``source`` and
        ``location`` are given.
    annotation : str, optional
        Short annotation to display under the caret, for example "undeclared variable". Omitted by default.
    notes : list of str, optional
        Hints to display at the bottom of the message. Empty by default.
    filepath : str, optional
        The file path to display in the error location. Defaults to ``<source>``.
    """

    error_code: str | None = None

    def __init__(
        self,
        message: str,
        source: str | None = None,
        location: "ParseLocation | None" = None,
        annotation: str | None = None,
        notes: list[str] | None = None,
        filepath: str | None = None,
    ):
        self.base_message = message
        self.source = source
        self.location = location
        self.annotation = annotation
        self.notes = notes or []
        self.filepath = filepath

        formatted_message = self._format_message()
        super().__init__(formatted_message)

    def _format_message(self) -> str:
        if self.source is None or self.location is None:
            return self.base_message

        return self._format_with_source()

    def _format_with_source(self) -> str:
        lines = self.source.splitlines()
        line = self.location.line

        if line < 1 or line > len(lines):
            return self.base_message

        header = f"error[{self.error_code}]: {self.base_message}" if self.error_code else self.base_message
        filepath = self.filepath or "<source>"
        column = self.location.column or 1
        parts = [header, f"  --> {filepath}:{line}:{column}", "     |"]

        start = max(0, line - 2)
        end = min(len(lines), line + 2)
        width = len(str(end))

        for i in range(start, end):
            line_num = i + 1
            line_content = lines[i].rstrip()
            parts.append(f"  {line_num:>{width}} | {line_content}".rstrip())

            if line_num == line and self.location.column is not None:
                parts.append(self._caret_line(width))

        parts.append("     |")
        parts.extend(f"   = note: {note}" for note in self.notes)

        return "\n".join(parts)

    def _caret_line(self, width: int) -> str:
        col_offset = self.location.column - 1
        if self.location.end_column is not None:
            pointer_len = max(1, self.location.end_column - self.location.column)
        else:
            pointer_len = 3

        gutter_width = width + len("  ") + len(" | ")
        padding = " " * (gutter_width + col_offset)
        caret = "^" * pointer_len

        if self.annotation:
            return f"{padding}{caret} {self.annotation}"
        return f"{padding}{caret}"


class DynamicCalibratingEquationException(GCNValidationError):
    """Raised when a calibrating equation contains variables with non-steady-state time indices."""

    error_code = "V001"

    def __init__(
        self,
        eq: sp.Add,
        block_name: str,
        source: str | None = None,
        location: "ParseLocation | None" = None,
        filepath: str | None = None,
    ):
        self.eq = eq
        self.block_name = block_name

        message = f"Calibrating equation in block {block_name} uses non-steady-state variables"

        super().__init__(
            message,
            source=source,
            location=location,
            annotation="variables must use [ss] time index",
            notes=["Write every variable in a calibrating equation with the [ss] time index: X[ss]"],
            filepath=filepath,
        )


class OptimizationProblemNotDefinedException(ValueError):
    """Raised when a block declares controls without an objective, or an objective without controls."""

    def __init__(self, block_name: str, missing: str) -> None:
        self.block_name = block_name
        self.missing = missing
        not_missing = "objective" if missing == "controls" else "controls"

        message = (
            f"Block {block_name} has a {missing} component but no {not_missing} component. Add the missing "
            f"component, or remove the {missing} if this block has no optimization problem."
        )

        super().__init__(message)


class MultipleObjectiveFunctionsException(ValueError):
    """Raised when a block declares more than one objective function."""

    def __init__(self, block_name: str, eqs: list[sp.Expr]) -> None:
        self.block_name = block_name

        n_eqs = len(eqs)

        listed_eqs = "\n".join(str(eq) for eq in eqs)
        message = (
            f"Block {block_name} declares {n_eqs} objectives, but only one is supported:\n{listed_eqs}\n"
            "Simplify the objective to a single function."
        )

        super().__init__(message)


class ControlVariableNotFoundException(GCNValidationError):
    """Raised when a declared control variable is not found in the block's equations."""

    error_code = "V002"

    def __init__(
        self,
        block_name: str,
        control: TimeAwareSymbol,
        source: str | None = None,
        location: "ParseLocation | None" = None,
        filepath: str | None = None,
    ):
        self.block_name = block_name
        self.control = control

        message = f"Control variable '{control}' in block {block_name} not found in equations"

        super().__init__(
            message,
            source=source,
            location=location,
            annotation="undeclared control variable",
            notes=[
                "Use the control in the objective or a constraint, with the same name and time index as declared",
                "Remove the control from the controls component",
            ],
            filepath=filepath,
        )


class ModelUnknownParameterError(ValueError):
    """Raised when a parameter update names a parameter that the model does not have."""

    def __init__(self, unknown_updates: list[str]):
        self.unknown_updates = unknown_updates

        message = (
            f"The following parameters were given new values, but do not exist in the model: "
            f"{', '.join(unknown_updates)}."
        )

        super().__init__(message)


class PerturbationSolutionNotFoundException(ValueError):
    """Raised when an operation needs a perturbation solution but the model has not been solved."""

    def __init__(self):
        message = (
            "This operation requires a perturbation solution. Call the .solve() method to solve for the policy "
            "function."
        )

        super().__init__(message)


class SteadyStateNotFoundError(ValueError):
    """Raised when the provided steady-state values leave non-zero residuals in some model equations."""

    def __init__(self, equations: list[str]):
        message = (
            "The provided steady-state values did not result in zero residuals for the following equations:\n"
            f"{', '.join(equations)}\n\nIf you used custom parameter values to compute the provided steady state, "
            f"you must also provide these parameter values to ``solve_model``."
        )

        super().__init__(message)


class GensysFailedException(ValueError):
    """Raised when the gensys solver cannot return a unique stable solution."""

    def __init__(self, eu: list[int] | tuple[int, ...]):
        message = interpret_gensys_output(eu)
        super().__init__(message)


class VariableNotFoundException(ValueError):
    """Raised when a requested variable is not among the model variables."""

    def __init__(self, variable: TimeAwareSymbol):
        message = f"Variable {variable.base_name} was not found among model variables."

        super().__init__(message)


class InvalidDistributionException(ValueError):
    """Raised when a distribution declaration in a GCN file cannot be interpreted."""

    def __init__(self, variable: str, distribution_string: str):
        message = (
            f'The distribution for "{variable}", defined as "{distribution_string}", could not be parsed. A shock '
            f"distribution takes no initial value, so remove any '= value' after it, as in "
            f"'epsilon[] ~ Normal(mu=0, sigma=1) = 0.5'."
        )

        super().__init__(message)


class MultipleParameterDefinitionException(ValueError):
    """Raised when a distribution declaration sets the same parameter more than once."""

    def __init__(self, variable_name: str, d_name: str, param_name: str, result_list: list[str]) -> None:
        message = (
            f'The {d_name} distribution for "{variable_name}" has multiple declarations for '
            f"{param_name}. Pass only one of: {', '.join(result_list)}"
        )

        super().__init__(message)


class InvalidParameterException(ValueError):
    """Raised when a distribution declaration passes a parameter the distribution does not accept."""

    def __init__(self, dist_name: str, param_name: str, valid_params: list[str]):
        message = (
            f"Unknown parameter {param_name} passed to distribution {dist_name}. Valid "
            f"parameters for this distribution are: {', '.join(valid_params)}"
        )

        super().__init__(message)


class OrphanParameterError(ValueError):
    """Raised when a parameter appears in the model equations but in no calibration block."""

    def __init__(self, orphans: Iterable[sp.Symbol]):
        orphans = set(orphans)
        n = len(orphans)
        verb = "was" if n == 1 else "were"
        message = (
            f"The following parameter{'s' if n > 1 else ''} {verb} found among model equations but did not appear in "
            f"any calibration block: {', '.join(x.name for x in orphans)}"
        )

        super().__init__(message)


def _extra_parameter_message(extras: Sequence[sp.Symbol]) -> str:
    n = len(extras)
    verb = "was" if n == 1 else "were"
    pronoun = "it" if n == 1 else "them"

    return (
        f"The following parameter{'s' if n > 1 else ''} {verb} given initial values in calibration blocks but "
        f"{verb} not used in model equations: {', '.join(x.name for x in extras)}. Delete {pronoun} from the "
        f"calibration block, or fix the equation that should use {pronoun}."
    )


class ExtraParameterError(ValueError):
    """Raised when a calibration block defines a parameter that no model equation uses."""

    def __init__(self, extras: Sequence[sp.Symbol]):
        super().__init__(_extra_parameter_message(extras))


class ExtraParameterWarning(UserWarning):
    """Warns that a calibration block defines a parameter that no model equation uses."""

    def __init__(self, extras: Sequence[sp.Symbol]):
        super().__init__(_extra_parameter_message(extras))


class DuplicateParameterError(GCNValidationError):
    """Raised when a parameter is defined more than once in calibration blocks."""

    error_code = "V003"

    def __init__(
        self,
        extras: Iterable[sp.Symbol],
        block: str | None = None,
        source: str | None = None,
        location: "ParseLocation | None" = None,
        filepath: str | None = None,
    ):
        param_names = ", ".join(x.name for x in extras)
        block_str = f"block {block}" if block else "calibration blocks"

        message = f"Duplicate parameter declaration in {block_str}"

        super().__init__(
            message,
            source=source,
            location=location,
            annotation=f"'{param_names}' already defined",
            notes=["Delete one of the declarations"],
            filepath=filepath,
        )
