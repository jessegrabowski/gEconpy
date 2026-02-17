from dataclasses import dataclass
from enum import Enum

import pyparsing as pp

from gEconpy.parser.error_catalog import ErrorCode


class Severity(Enum):
    """Severity level for parse errors."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class GCNParseFailure(pp.ParseFatalException):
    """
    Custom parse exception that carries structured error data.

    This extends pyparsing's ParseFatalException to include:
    - An ErrorCode from our catalog
    - The found token/content for better error messages
    - Optional suggestions for typo correction

    Data is encoded in the message to survive pyparsing's exception wrapping.
    Format: "message||CODE||found||suggestion1,suggestion2"
    """

    SEPARATOR = "||GCN||"

    def __init__(
        self,
        s: str,
        loc: int = 0,
        msg: str = "",
        code: ErrorCode = ErrorCode.E000,
        found: str = "",
        suggestions: list[str] | None = None,
    ):
        suggestions_list = suggestions or []
        suggestions_str = ",".join(suggestions_list)
        encoded_msg = self.SEPARATOR.join((msg, code.name, found, suggestions_str))
        super().__init__(s, loc, encoded_msg)

        # Also store as attributes for direct access if exception isn't wrapped
        self.error_code = code
        self.gcn_found = found
        self.suggestions = suggestions_list

    @classmethod
    def decode(cls, exc: pp.ParseBaseException) -> tuple[str, ErrorCode, str, list[str]]:
        """
        Decode error data from an exception message.

        Returns (message, code, found, suggestions).
        If not encoded, returns (original_msg, E000, "", []).
        """
        msg = str(exc.msg) if hasattr(exc, "msg") else str(exc)

        if cls.SEPARATOR not in msg:
            found = exc.found if hasattr(exc, "found") and exc.found else ""
            return msg, ErrorCode.E000, found, []

        parts = msg.split(cls.SEPARATOR)
        expected_parts = 4  # message, code, found, suggestions
        if len(parts) != expected_parts:
            return msg, ErrorCode.E000, "", []

        message, code_str, found, suggestions_str = parts
        try:
            code = ErrorCode[code_str]
        except KeyError:
            code = ErrorCode.E000
        suggestions = [s for s in suggestions_str.split(",") if s]
        return message, code, found, suggestions

    def copy(self) -> "GCNParseFailure":
        """Create a copy of this exception (used by pyparsing caching)."""
        return GCNParseFailure(
            self.pstr,
            self.loc,
            # Extract original message (without encoding)
            self.msg.split(self.SEPARATOR)[0] if self.SEPARATOR in self.msg else self.msg,
            code=self.error_code,
            found=self.gcn_found,
            suggestions=self.suggestions,
        )


@dataclass(frozen=True)
class ParseLocation:
    """
    Track the location of a token or error in the source text.

    Attributes
    ----------
    line : int
        The 1-based line number where the error occurred.
    column : int
        The 1-based column number where the error occurred.
    end_line : int, optional
        The 1-based line number where the error ends (for span highlighting).
    end_column : int, optional
        The 1-based column number where the error ends.
    source_line : str, optional
        The full text of the line where the error occurred.
    filename : str, optional
        The name of the file being parsed.

    Examples
    --------
    .. testcode::

        from gEconpy.parser.errors import ParseLocation

        loc = ParseLocation(line=5, column=12, source_line="    Y[] = C[] + I[];")
        print(loc.format_pointer())

    .. testoutput::

            Y[] = C[] + I[];
                   ^
    """

    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None
    source_line: str = ""
    filename: str = ""

    def to_lsp_range(self) -> dict:
        """Convert to LSP Range format (0-indexed)."""
        end_line = self.end_line if self.end_line is not None else self.line
        end_col = self.end_column if self.end_column is not None else self.column + 1
        return {
            "start": {"line": self.line - 1, "character": self.column - 1},
            "end": {"line": end_line - 1, "character": end_col - 1},
        }

    def format_pointer(self, pointer_char: str = "^") -> str:
        """
        Format the source line with a pointer to the error column.

        Parameters
        ----------
        pointer_char : str, optional
            Character to use for the pointer. Default is "^".

        Returns
        -------
        formatted_pointer : str
            The source line followed by a pointer line.
        """
        if not self.source_line:
            return ""

        col_index = max(0, self.column - 1)
        if self.end_column is not None and self.end_line == self.line:
            pointer_len = max(1, self.end_column - self.column)
            pointer_line = " " * col_index + pointer_char * pointer_len
        else:
            pointer_line = " " * col_index + pointer_char
        return f"{self.source_line}\n{pointer_line}"

    def format_location(self) -> str:
        """
        Format the location as a string.

        Returns
        -------
        location : str
            A string like "file.gcn:5:12" or "line 5, column 12".
        """
        if self.filename:
            return f"{self.filename}:{self.line}:{self.column}"
        return f"line {self.line}, column {self.column}"

    def __str__(self) -> str:
        return self.format_location()


class GCNParseError(Exception):
    """
    Base class for all GCN parsing errors.

    This provides a consistent interface for errors with optional source
    location tracking and suggestions for fixes.

    Parameters
    ----------
    message : str
        The main error message.
    location : ParseLocation, optional
        The location in the source where the error occurred.
    suggestions : list of str, optional
        Suggested fixes or alternatives (shown as "help: Did you mean...").
    context : str, optional
        Additional context about the error (e.g., block name).
    code : ErrorCode, optional
        Error code (e.g., "E001", "W002").
    severity : Severity, optional
        Error severity ("error" or "warning"). Default is "error".
    annotation : str, optional
        Short explanation shown after the caret pointer (e.g., "undefined parameter").
    notes : list of str, optional
        Additional notes shown after the error (e.g., "Parameters must be defined...").

    Examples
    --------
    .. testcode::

        from gEconpy.parser.errors import GCNParseError, ParseLocation

        loc = ParseLocation(line=10, column=5, source_line="    alpha ~ Beta(mena=0.5);")
        err = GCNParseError(
            "Unknown parameter 'mena'",
            location=loc,
            suggestions=["mean"]
        )
        print(err)

    .. testoutput::

        Unknown parameter 'mena'
          at line 10, column 5
                alpha ~ Beta(mena=0.5);
                ^
          Did you mean: mean?
    """

    def __init__(
        self,
        message: str,
        location: ParseLocation | None = None,
        suggestions: list[str] | None = None,
        context: str = "",
        code: ErrorCode | None = None,
        severity: Severity = Severity.ERROR,
        annotation: str = "",
        notes: list[str] | None = None,
    ):
        self.message = message
        self.location = location
        self.suggestions = suggestions or []
        self.context = context
        self.code = code
        self.severity = severity
        self.annotation = annotation
        self.notes = notes or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the complete error message with location and suggestions."""
        parts = [f"[{self.code.name}] {self.message}"] if self.code else [self.message]

        if self.context:
            parts.append(f"  in {self.context}")

        if self.location:
            parts.append(f"  at {self.location.format_location()}")
            pointer = self.location.format_pointer()
            if pointer:
                # Indent the pointer block
                indented = "\n".join("    " + line for line in pointer.split("\n"))
                parts.append(indented)

        if self.suggestions:
            if len(self.suggestions) == 1:
                parts.append(f"  Did you mean: {self.suggestions[0]}?")
            else:
                parts.append(f"  Did you mean one of: {', '.join(self.suggestions)}?")

        return "\n".join(parts)

    def to_lsp_diagnostic(self) -> dict:
        """
        Convert to LSP Diagnostic format for editor integration.

        Returns a dictionary conforming to the Language Server Protocol
        Diagnostic specification.
        """
        severity_map = {
            Severity.ERROR: 1,
            Severity.WARNING: 2,
            Severity.INFO: 3,
            Severity.HINT: 4,
        }

        if self.location:
            range_dict = self.location.to_lsp_range()
        else:
            range_dict = {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 0},
            }

        diagnostic: dict = {
            "range": range_dict,
            "message": self.message,
            "severity": severity_map.get(self.severity, 1),
            "source": "gEconpy",
        }

        if self.code:
            diagnostic["code"] = self.code.name
            diagnostic["codeDescription"] = {"href": f"https://geconpy.readthedocs.io/errors/{self.code.name}.html"}

        if self.suggestions:
            diagnostic["data"] = {"suggestions": self.suggestions}

        return diagnostic

    def with_location(self, location: ParseLocation) -> "GCNParseError":
        """
        Create a new error with the given location.

        This is useful when location information becomes available after
        the error is initially created.

        Parameters
        ----------
        location : ParseLocation
            The source location to add.

        Returns
        -------
        error : GCNParseError
            A new error instance with the location set.
        """
        return self.__class__(
            message=self.message,
            location=location,
            suggestions=self.suggestions,
            context=self.context,
            code=self.code,
            severity=self.severity,
            annotation=self.annotation,
            notes=self.notes,
        )

    def with_context(self, context: str) -> "GCNParseError":
        """
        Create a new error with additional context.

        Parameters
        ----------
        context : str
            Context to add (e.g., "block HOUSEHOLD").

        Returns
        -------
        error : GCNParseError
            A new error instance with the context set.
        """
        return self.__class__(
            message=self.message,
            location=self.location,
            suggestions=self.suggestions,
            context=context,
            code=self.code,
            severity=self.severity,
            annotation=self.annotation,
            notes=self.notes,
        )


class GCNGrammarError(GCNParseError):
    """
    Error in the grammatical structure of GCN source.

    Raised when the parser encounters valid tokens in an invalid arrangement,
    such as missing semicolons, unbalanced braces, or unexpected tokens.

    Parameters
    ----------
    message : str
        Description of the grammar error.
    expected : str or list of str, optional
        What the parser expected to find.
    found : str, optional
        What was actually found.
    location : ParseLocation, optional
        Where in the source the error occurred.

    Examples
    --------
    .. testcode::

        from gEconpy.parser.errors import GCNGrammarError, ParseLocation

        err = GCNGrammarError(
            "Missing semicolon",
            expected=";",
            found="}",
            location=ParseLocation(line=20, column=1)
        )
    """

    def __init__(
        self,
        message: str,
        expected: str | list[str] | None = None,
        found: str = "",
        location: ParseLocation | None = None,
        context: str = "",
        code: ErrorCode = ErrorCode.E000,
        severity: Severity = Severity.ERROR,
        annotation: str = "",
        notes: list[str] | None = None,
        suggestions: list[str] | None = None,
    ):
        self.expected = expected if isinstance(expected, list) else ([expected] if expected else [])
        self.found = found

        # Enhance message with expected/found info
        if self.expected and self.found:
            if len(self.expected) == 1:
                message = f"{message}. Expected '{self.expected[0]}', found '{self.found}'"
            else:
                expected_str = ", ".join(f"'{e}'" for e in self.expected)
                message = f"{message}. Expected one of {expected_str}, found '{self.found}'"
        elif self.expected:
            if len(self.expected) == 1:
                message = f"{message}. Expected '{self.expected[0]}'"
            else:
                expected_str = ", ".join(f"'{e}'" for e in self.expected)
                message = f"{message}. Expected one of {expected_str}"
        elif self.found:
            message = f"{message}. Found '{self.found}'"

        super().__init__(
            message=message,
            location=location,
            suggestions=suggestions,
            context=context,
            code=code,
            severity=severity,
            annotation=annotation,
            notes=notes,
        )


class GCNSemanticError(GCNParseError):
    """
    Error in the semantics/meaning of GCN source.

    Raised when the source is syntactically valid but semantically incorrect,
    such as referencing undefined variables or using invalid time indices.

    Parameters
    ----------
    message : str
        Description of the semantic error.
    symbol_name : str, optional
        The name of the problematic symbol.
    location : ParseLocation, optional
        Where in the source the error occurred.
    suggestions : list of str, optional
        Suggested fixes (e.g., similar variable names).

    Examples
    --------
    .. testcode::

        from gEconpy.parser.errors import GCNSemanticError, ParseLocation

        err = GCNSemanticError(
            "Undefined variable",
            symbol_name="Consumptin",
            suggestions=["Consumption"],
            location=ParseLocation(line=15, column=8)
        )
    """

    def __init__(
        self,
        message: str,
        symbol_name: str = "",
        location: ParseLocation | None = None,
        suggestions: list[str] | None = None,
        context: str = "",
        code: ErrorCode | None = None,
        severity: Severity = Severity.ERROR,
    ):
        self.symbol_name = symbol_name
        if symbol_name and symbol_name not in message:
            message = f"{message}: '{symbol_name}'"
        super().__init__(
            message=message,
            location=location,
            suggestions=suggestions,
            context=context,
            code=code,
            severity=severity,
        )


class GCNErrorCollection(Exception):
    """
    A collection of multiple parse/validation errors.

    This exception is raised when multiple errors are collected during
    parsing or validation, allowing all errors to be reported at once
    rather than stopping at the first error.

    Parameters
    ----------
    errors : list of GCNParseError
        The collected errors.
    source : str, optional
        The source text being parsed (for context).

    Examples
    --------
    .. testcode::

        from gEconpy.parser.errors import GCNErrorCollection, GCNSemanticError, ParseLocation

        errors = [
            GCNSemanticError("Undefined variable 'X'", location=ParseLocation(1, 5)),
            GCNSemanticError("Undefined variable 'Y'", location=ParseLocation(2, 5)),
        ]
        exc = GCNErrorCollection(errors)
        print(len(exc))

    .. testoutput::

        2
    """

    def __init__(
        self,
        errors: list[GCNParseError],
        source: str | None = None,
    ):
        self.errors = errors
        self.source = source
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format all errors with separators."""
        if not self.errors:
            return "No errors"

        if len(self.errors) == 1:
            return str(self.errors[0])

        parts = [f"Found {len(self.errors)} errors:"]
        for i, error in enumerate(self.errors, 1):
            parts.append(f"\n[{i}] {error}")

        return "".join(parts)

    def __len__(self) -> int:
        return len(self.errors)

    def __iter__(self):
        return iter(self.errors)

    def __getitem__(self, index: int) -> GCNParseError:
        return self.errors[index]

    @property
    def has_errors(self) -> bool:
        """Return True if there are any errors."""
        return len(self.errors) > 0

    def to_lsp_diagnostics(self) -> list[dict]:
        """Convert all errors to LSP Diagnostic format."""
        return [err.to_lsp_diagnostic() for err in self.errors]


class ErrorCollector:
    """
    Helper class for collecting multiple errors during parsing.

    Use this to accumulate errors during a parsing pass and then
    raise them all at once.

    Parameters
    ----------
    source : str, optional
        The source text being parsed.

    Examples
    --------
    .. testcode::

        from gEconpy.parser.errors import ErrorCollector, GCNSemanticError

        collector = ErrorCollector()
        collector.add(GCNSemanticError("Error 1"))
        collector.add(GCNSemanticError("Error 2"))
        print(len(collector))

    .. testoutput::

        2
    """

    def __init__(self, source: str | None = None):
        self.errors: list[GCNParseError] = []
        self.source = source

    def add(self, error: GCNParseError) -> None:
        """Add an error to the collection."""
        self.errors.append(error)

    def raise_if_errors(self) -> None:
        """Raise GCNErrorCollection if there are any errors."""
        if self.errors:
            raise GCNErrorCollection(self.errors, self.source)

    def raise_first(self) -> None:
        """Raise the first error as an exception."""
        for error in self.errors:
            if getattr(error, "severity", Severity.ERROR) == Severity.ERROR:
                raise error

    @property
    def has_errors(self) -> bool:
        """Return True if there are any error-level issues."""
        return any(getattr(e, "severity", Severity.ERROR) == Severity.ERROR for e in self.errors)

    @property
    def warnings(self) -> list[GCNParseError]:
        """Return all warning-level issues."""
        return [e for e in self.errors if getattr(e, "severity", Severity.ERROR) == Severity.WARNING]

    def __len__(self) -> int:
        return len(self.errors)

    def __bool__(self) -> bool:
        return len(self.errors) > 0

    def __iter__(self):
        return iter(self.errors)
