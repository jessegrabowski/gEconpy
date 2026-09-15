from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.solvers.gensys import interpret_gensys_output

if TYPE_CHECKING:
    from gEconpy.parser.errors import ParseLocation


class GCNValidationError(ValueError):
    """
    Base class for validation errors that support rich source location display.

    This provides a consistent interface for errors that occur during model validation
    (post-parsing) and can display the relevant source code with highlighted tokens.

    Parameters
    ----------
    message : str
        The main error message.
    source : str, optional
        The full source code of the GCN file.
    location : ParseLocation, optional
        The location of the problematic token in the source.
    annotation : str, optional
        Short annotation to display under the caret (e.g., "undeclared variable").
    notes : list of str, optional
        Additional notes/hints to display at the bottom.
    filepath : str, optional
        The file path to display in the error location.

    Examples
    --------
    .. code-block:: python

        from gEconpy.exceptions import GCNValidationError
        from gEconpy.parser.errors import ParseLocation

        loc = ParseLocation(line=5, column=12, end_column=15, source_line="    K_t[];")
        err = GCNValidationError(
            "Variable K_t not found in equations",
            source=source_code,
            location=loc,
            annotation="undeclared control",
            notes=["Check that the variable appears in constraints or objective"],
        )
    """

    # Subclasses can override this for specific error codes
    error_code: str | None = None

    def __init__(
        self,
        message: str,
        source: str | None = None,
        location: ParseLocation | None = None,
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
        """Format the error message with source context if available."""
        if self.source is None or self.location is None:
            return self.base_message

        return self._format_with_source()

    def _format_with_source(self) -> str:
        """Format the error message in the style of parsing errors."""
        lines = self.source.splitlines()
        line = self.location.line

        if line < 1 or line > len(lines):
            return self.base_message

        parts = []

        # Error header: error[CODE]: message
        if self.error_code:
            parts.append(f"error[{self.error_code}]: {self.base_message}")
        else:
            parts.append(self.base_message)

        # File location: --> file:line:column
        filepath = self.filepath or "<source>"
        col = self.location.column or 1
        parts.append(f"  --> {filepath}:{line}:{col}")

        # Empty line with just pipe
        parts.append("     |")

        # Context lines (before, error line, after)
        start = max(0, line - 2)
        end = min(len(lines), line + 2)

        # Calculate width needed for line numbers
        width = len(str(end))

        for i in range(start, end):
            line_num = i + 1
            line_content = lines[i].rstrip()
            if line_content:
                parts.append(f"  {line_num:>{width}} | {line_content}")
            else:
                parts.append(f"  {line_num:>{width}} |")

            # Add caret and annotation under the error line
            if line_num == line and self.location.column is not None:
                # Calculate pointer position
                col_offset = self.location.column - 1
                if self.location.end_column is not None:
                    pointer_len = max(1, self.location.end_column - self.location.column)
                else:
                    pointer_len = 3  # Default to 3 carets

                padding = " " * (width + 5 + col_offset)  # 5 = "  " + " | "
                caret = "^" * pointer_len

                if self.annotation:
                    parts.append(f"{padding}{caret} {self.annotation}")
                else:
                    parts.append(f"{padding}{caret}")

        # Closing pipe
        parts.append("     |")

        # Notes at the bottom
        parts.extend(f"   = note: {note}" for note in self.notes)

        return "\n".join(parts)


class DynamicCalibratingEquationException(GCNValidationError):
    """Raised when a calibrating equation contains variables with non-steady-state time indices."""

    error_code = "V001"

    def __init__(
        self,
        eq: sp.Add,
        block_name: str,
        source: str | None = None,
        location: ParseLocation | None = None,
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
            notes=[
                "Calibrating equations define steady-state relationships",
                "Use X[ss] instead of X[] for all variables",
            ],
            filepath=filepath,
        )


class OptimizationProblemNotDefinedException(ValueError):
    """Raised when a block declares controls without an objective, or an objective without controls."""

    def __init__(self, block_name: str, missing: str) -> None:
        self.block_name = block_name
        self.missing = missing
        not_missing = "objective" if missing == "controls" else "controls"

        message = (
            f"Block {block_name} has a {missing} component but no {not_missing} component, verify whether"
            f"or not this block has an optimization problem."
        )

        super().__init__(message)


class MultipleObjectiveFunctionsException(ValueError):
    """Raised when a block declares more than one objective function."""

    def __init__(self, block_name: str, eqs: list[sp.Expr]) -> None:
        self.block_name = block_name

        n_eqs = len(eqs)

        message = f"Block {block_name} appears to have multiple objectives, excepted just one but found {n_eqs}:\n"
        for eq in eqs:
            message += str(eq) + "\n"
        message += (
            " Only one objective function is supported. Please manually simplify the objective to a single function."
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
        location: ParseLocation | None = None,
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
                "Control variables must appear in the objective or constraints",
                "Check spelling of the variable name",
                "Verify the variable has the correct time index",
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
            "This operation cannot be completed until the model has a solved perturbation solution. Please "
            "call the .solve() method to solve for the policy function."
        )

        super().__init__(message)


class SteadyStateNotFoundError(ValueError):
    """Raised when the provided steady-state values leave non-zero residuals in some model equations."""

    def __init__(self, equations):
        message = (
            "The provided steady-state values did not result in zero residuals for the following equations:\n"
            f"{', '.join(equations)}\n\nIf you used custom parameter values to compute the provided steady state, "
            f"you must also provide these parameter values to ``solve_model``."
        )

        super().__init__(message)


class GensysFailedException(ValueError):
    """Raised when the gensys solver cannot return a unique stable solution."""

    def __init__(self, eu):
        message = interpret_gensys_output(eu)
        super().__init__(message)


class VariableNotFoundException(ValueError):
    """Raised when a requested variable is not among the model variables."""

    def __init__(self, variable):
        var_name = variable.base_name
        message = f"Variable {var_name} was not found among model variables."

        super().__init__(message)


class InvalidDistributionException(ValueError):
    """Raised when a distribution declaration in a GCN file cannot be interpreted."""

    def __init__(self, variable, distribution_string):
        message = (
            f'The distribution associated with "{variable}", defined as "{distribution_string}", appears to have '
            f"a typo, please check the GCN file. Please also check that you have not supplied an initial "
            f"parameter value to an exogenous shock distribution, as in epsilon[] ~ N(mu=0, sd=1) = 0.5. Shock "
            f"distributions should NOT have an equals sign after the distribution definition."
        )

        super().__init__(message)


class MultipleParameterDefinitionException(ValueError):
    """Raised when a distribution declaration sets the same parameter more than once."""

    def __init__(self, variable_name: str, d_name: str, param_name: str, result_list: list[str]) -> None:
        message = (
            f'The {d_name} distribution associated with "{variable_name}" has multiple declarations for '
            f"{param_name}. Please pass only one of: "
        )
        message += ", ".join(result_list)

        super().__init__(message)


class InvalidParameterException(ValueError):
    """Raised when a distribution declaration passes a parameter the distribution does not accept."""

    def __init__(self, dist_name, param_name, valid_params):
        message = (
            f"Unknown parameter {param_name} passed to distribution {dist_name}. Valid "
            f"parameters for this distribution are: {', '.join(valid_params)}"
        )

        super().__init__(message)


class OrphanParameterError(ValueError):
    """Raised when a parameter appears in the model equations but in no calibration block."""

    def __init__(self, orphans):
        orphans = set(orphans)
        n = len(orphans)
        verb = "was" if n == 1 else "were"
        message = (
            f"The following parameter{'s' if n > 1 else ''} {verb} found among model equations but did not appear in "
            f"any calibration block: {', '.join([x.name for x in orphans])}"
        )

        super().__init__(message)


class ExtraParameterError(ValueError):
    """Raised when a calibration block defines a parameter that no model equation uses."""

    def __init__(self, extras):
        n = len(extras)
        verb = "was" if n == 1 else "were"
        message = (
            f"The following parameter{'s' if n > 1 else ''} {verb} were given initial values in calibration blocks but "
            f"were not used in model equations: {', '.join([x.name for x in extras])} \n"
            f"Verify your model equations, or remove these parameters if they are not needed."
        )

        super().__init__(message)


class ExtraParameterWarning(UserWarning):
    """Warns that a calibration block defines a parameter that no model equation uses."""

    def __init__(self, extras):
        n = len(extras)
        verb = "was" if n == 1 else "were"
        message = (
            f"The following parameter{'s' if n > 1 else ''} {verb} were given initial values in calibration blocks but "
            f"were not used in model equations: {', '.join([x.name for x in extras])} \n"
            f"Verify your model equations, or remove these parameters if they are not needed."
        )

        super().__init__(message)


class DuplicateParameterError(GCNValidationError):
    """Raised when a parameter is defined more than once in calibration blocks."""

    error_code = "V003"

    def __init__(
        self,
        extras,
        block: str | None = None,
        source: str | None = None,
        location: ParseLocation | None = None,
        filepath: str | None = None,
    ):
        len(extras)
        param_names = ", ".join([x.name for x in extras])
        block_str = f"block {block}" if block else "calibration blocks"

        message = f"Duplicate parameter declaration in {block_str}"

        super().__init__(
            message,
            source=source,
            location=location,
            annotation=f"'{param_names}' already defined",
            notes=[
                "Each parameter should be declared only once",
                "Remove the duplicate declaration",
            ],
            filepath=filepath,
        )
