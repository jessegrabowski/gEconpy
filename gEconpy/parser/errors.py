from dataclasses import dataclass
from enum import Enum

import pyparsing as pp

from gEconpy.parser.error_catalog import ErrorCode


class Severity(Enum):
    """Severity of a parse error, ordered from most to least serious."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


_LSP_SEVERITY = {
    Severity.ERROR: 1,
    Severity.WARNING: 2,
    Severity.INFO: 3,
    Severity.HINT: 4,
}


class GCNParseFailure(pp.ParseFatalException):
    """
    Fatal pyparsing exception carrying an :class:`ErrorCode`, the offending text, and typo suggestions.

    pyparsing wraps and re-raises exceptions while backtracking, and only the message survives that wrapping. The
    structured data is therefore encoded into the message as ``message||GCN||CODE||GCN||found||GCN||s1,s2`` and
    recovered with :meth:`decode`.

    Parameters
    ----------
    s : str
        The full source text being parsed.
    loc : int, optional
        Character offset of the failure in ``s``. Defaults to 0.
    msg : str, optional
        The error message. Defaults to an empty string.
    code : ErrorCode, optional
        The catalog code for the failure. Defaults to ``ErrorCode.E000``.
    found : str, optional
        The text that triggered the failure. Defaults to an empty string.
    suggestions : list of str, optional
        Names the user may have meant. Defaults to no suggestions.
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
        encoded_msg = self.SEPARATOR.join((msg, code.name, found, ",".join(suggestions_list)))
        super().__init__(s, loc, encoded_msg)

        self.error_code = code
        self.gcn_found = found
        self.suggestions = suggestions_list

    @classmethod
    def decode(cls, exc: pp.ParseBaseException) -> tuple[str, ErrorCode, str, list[str]]:
        """
        Recover the structured data packed into an exception message by this class.

        Parameters
        ----------
        exc : ParseBaseException
            The pyparsing exception to decode. An exception without encoded data decodes to code ``E000``.

        Returns
        -------
        message : str
            The error message.
        code : ErrorCode
            The error code.
        found : str
            The text that triggered the error.
        suggestions : list of str
            Names the user may have meant.
        """
        msg = str(exc.msg) if hasattr(exc, "msg") else str(exc)

        if cls.SEPARATOR not in msg:
            found = exc.found if hasattr(exc, "found") and exc.found else ""
            return msg, ErrorCode.E000, found, []

        parts = msg.split(cls.SEPARATOR)
        expected_parts = 4
        if len(parts) != expected_parts:
            return msg, ErrorCode.E000, "", []

        message, code_name, found, suggestions_str = parts
        try:
            code = ErrorCode[code_name]
        except KeyError:
            code = ErrorCode.E000
        suggestions = [s for s in suggestions_str.split(",") if s]

        return message, code, found, suggestions

    def copy(self) -> "GCNParseFailure":
        """
        Copy this exception, as pyparsing's packrat cache requires.

        Returns
        -------
        exception : GCNParseFailure
            A copy carrying the same code, found text, and suggestions.
        """
        original_message = self.msg.split(self.SEPARATOR)[0]
        return GCNParseFailure(
            self.pstr,
            self.loc,
            original_message,
            code=self.error_code,
            found=self.gcn_found,
            suggestions=self.suggestions,
        )


@dataclass(frozen=True)
class ParseLocation:
    """
    Position of a token or error in the source text.

    Parameters
    ----------
    line : int
        1-based line number.
    column : int
        1-based column number.
    end_line : int, optional
        1-based line number where the span ends. Defaults to None, meaning a single-character span.
    end_column : int, optional
        1-based column number where the span ends. Defaults to None, meaning a single-character span.
    source_line : str, optional
        Full text of the line at ``line``. Defaults to an empty string.
    filename : str, optional
        Name of the file being parsed. Defaults to an empty string.

    Examples
    --------
    Point at a column of a source line:

    .. code-block:: python

        from gEconpy.parser.errors import ParseLocation

        loc = ParseLocation(line=5, column=12, source_line="    Y[] = C[] + I[];")
        print(loc.format_pointer())
    """

    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None
    source_line: str = ""
    filename: str = ""

    def to_lsp_range(self) -> dict:
        """
        Convert this location to a Language Server Protocol range.

        Returns
        -------
        lsp_range : dict
            A dictionary with ``start`` and ``end`` keys, each holding zero-indexed ``line`` and ``character``.
        """
        end_line = self.end_line if self.end_line is not None else self.line
        end_column = self.end_column if self.end_column is not None else self.column + 1
        return {
            "start": {"line": self.line - 1, "character": self.column - 1},
            "end": {"line": end_line - 1, "character": end_column - 1},
        }

    def format_pointer(self, pointer_char: str = "^") -> str:
        """
        Render the source line with a pointer underneath the error span.

        Parameters
        ----------
        pointer_char : str, optional
            Character repeated to draw the pointer. Defaults to ``"^"``.

        Returns
        -------
        formatted_pointer : str
            The source line followed by the pointer line, or an empty string when no source line is known.
        """
        if not self.source_line:
            return ""

        pointer_length = 1
        if self.end_column is not None and self.end_line == self.line:
            pointer_length = max(1, self.end_column - self.column)

        padding = " " * max(0, self.column - 1)
        return f"{self.source_line}\n{padding}{pointer_char * pointer_length}"

    def format_location(self) -> str:
        """
        Render the location as ``file.gcn:5:12`` or, without a filename, ``line 5, column 12``.

        Returns
        -------
        location : str
            The rendered location.
        """
        if self.filename:
            return f"{self.filename}:{self.line}:{self.column}"
        return f"line {self.line}, column {self.column}"

    def __str__(self) -> str:
        return self.format_location()


class GCNParseError(Exception):
    """
    Base class for all GCN parsing errors.

    Parameters
    ----------
    message : str
        The main error message.
    location : ParseLocation, optional
        Where in the source the error occurred. Defaults to None.
    suggestions : list of str, optional
        Names the user may have meant, rendered as "Did you mean ...". Defaults to no suggestions.
    context : str, optional
        Where the error was found, such as a block name. Defaults to an empty string.
    code : ErrorCode, optional
        The catalog code, rendered as a ``[E001]`` prefix. Defaults to None.
    severity : Severity, optional
        How serious the error is. Defaults to ``Severity.ERROR``.
    annotation : str, optional
        Short label rendered after the caret pointer, such as "undefined parameter". Defaults to an empty string.
    notes : list of str, optional
        Extra lines rendered after the error. Defaults to no notes.

    Examples
    --------
    Attach a location and a suggestion to an error, then print it:

    .. code-block:: python

        from gEconpy.parser.errors import GCNParseError, ParseLocation

        loc = ParseLocation(line=10, column=5, source_line="    alpha ~ Beta(mena=0.5);")
        err = GCNParseError("Unknown parameter 'mena'", location=loc, suggestions=["mean"])
        print(err)
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
        parts = [f"[{self.code.name}] {self.message}"] if self.code else [self.message]

        if self.context:
            parts.append(f"  in {self.context}")

        if self.location:
            parts.append(f"  at {self.location.format_location()}")
            pointer = self.location.format_pointer()
            if pointer:
                parts.append("\n".join("    " + line for line in pointer.split("\n")))

        if len(self.suggestions) == 1:
            parts.append(f"  Did you mean: {self.suggestions[0]}?")
        elif self.suggestions:
            parts.append(f"  Did you mean one of: {', '.join(self.suggestions)}?")

        return "\n".join(parts)

    def to_lsp_diagnostic(self) -> dict:
        """
        Convert this error to a Language Server Protocol diagnostic, for editor integration.

        Returns
        -------
        diagnostic : dict
            A dictionary conforming to the Language Server Protocol diagnostic specification.
        """
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
            "severity": _LSP_SEVERITY.get(self.severity, 1),
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
        Copy this error with a location attached, for when the location is only known after the error is raised.

        Parameters
        ----------
        location : ParseLocation
            The source location to attach.

        Returns
        -------
        error : GCNParseError
            A new error of the same class with the location set.
        """
        return self._copy_with(location=location)

    def with_context(self, context: str) -> "GCNParseError":
        """
        Copy this error with context attached.

        Parameters
        ----------
        context : str
            Where the error was found, such as ``"block HOUSEHOLD"``.

        Returns
        -------
        error : GCNParseError
            A new error of the same class with the context set.
        """
        return self._copy_with(context=context)

    def _copy_with(self, **changes) -> "GCNParseError":
        fields = {
            "message": self.message,
            "location": self.location,
            "suggestions": self.suggestions,
            "context": self.context,
            "code": self.code,
            "severity": self.severity,
            "annotation": self.annotation,
            "notes": self.notes,
        }
        return self.__class__(**{**fields, **changes})


class GCNGrammarError(GCNParseError):
    """
    Error in the grammatical structure of GCN source, such as a missing semicolon or an unbalanced brace.

    Parameters
    ----------
    message : str
        Description of the grammar error. When ``expected`` or ``found`` is given, the message gains a trailing
        sentence naming them.
    expected : str or list of str, optional
        What the parser expected to find. Defaults to nothing.
    found : str, optional
        What the parser found instead. Defaults to an empty string.
    location : ParseLocation, optional
        Where in the source the error occurred. Defaults to None.
    context : str, optional
        Where the error was found, such as a block name. Defaults to an empty string.
    code : ErrorCode, optional
        The catalog code. Defaults to ``ErrorCode.E000``.
    severity : Severity, optional
        How serious the error is. Defaults to ``Severity.ERROR``.
    annotation : str, optional
        Short label rendered after the caret pointer. Defaults to an empty string.
    notes : list of str, optional
        Extra lines rendered after the error. Defaults to no notes.
    suggestions : list of str, optional
        Names the user may have meant. Defaults to no suggestions.
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

        detail = _describe_expectation(self.expected, self.found)
        if detail:
            message = f"{message}. {detail}"

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


def _describe_expectation(expected: list[str], found: str) -> str:
    if len(expected) == 1:
        expected_text = f"Expected '{expected[0]}'"
    elif expected:
        expected_text = "Expected one of " + ", ".join(f"'{e}'" for e in expected)
    else:
        expected_text = ""

    if expected_text and found:
        return f"{expected_text}, found '{found}'"
    if expected_text:
        return expected_text
    if found:
        return f"Found '{found}'"
    return ""


class GCNSemanticError(GCNParseError):
    """
    Error in the meaning of syntactically valid GCN source, such as a reference to an undefined variable.

    Parameters
    ----------
    message : str
        Description of the semantic error.
    symbol_name : str, optional
        The offending symbol, appended to the message when the message does not already contain it. Defaults to an
        empty string.
    location : ParseLocation, optional
        Where in the source the error occurred. Defaults to None.
    suggestions : list of str, optional
        Names the user may have meant. Defaults to no suggestions.
    context : str, optional
        Where the error was found, such as a block name. Defaults to an empty string.
    code : ErrorCode, optional
        The catalog code. Defaults to None.
    severity : Severity, optional
        How serious the error is. Defaults to ``Severity.ERROR``.
    annotation : str, optional
        Short label rendered after the caret pointer. Defaults to an empty string.
    notes : list of str, optional
        Extra lines rendered after the error. Defaults to no notes.
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
        annotation: str = "",
        notes: list[str] | None = None,
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
            annotation=annotation,
            notes=notes,
        )


class GCNErrorCollection(Exception):
    """
    Several parse or validation errors raised together, so that one run reports every problem in the file.

    Parameters
    ----------
    errors : list of GCNParseError
        The collected errors.
    source : str, optional
        The source text being parsed, used by formatters to show context. Defaults to None.
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
        if not self.errors:
            return "No errors"

        if len(self.errors) == 1:
            return str(self.errors[0])

        parts = [f"Found {len(self.errors)} errors:"]
        parts.extend(f"\n[{i}] {error}" for i, error in enumerate(self.errors, 1))
        return "".join(parts)

    def __len__(self) -> int:
        return len(self.errors)

    def __iter__(self):
        return iter(self.errors)

    def __getitem__(self, index: int) -> GCNParseError:
        return self.errors[index]

    @property
    def has_errors(self) -> bool:
        """True if the collection is not empty."""
        return len(self.errors) > 0

    def to_lsp_diagnostics(self) -> list[dict]:
        """
        Convert every collected error to a Language Server Protocol diagnostic.

        Returns
        -------
        diagnostics : list of dict
            One diagnostic per error.
        """
        return [err.to_lsp_diagnostic() for err in self.errors]


class ErrorCollector:
    """
    Accumulate errors and warnings during a parsing pass, to raise them together afterwards.

    Parameters
    ----------
    source : str, optional
        The source text being parsed, passed on to the raised :class:`GCNErrorCollection`. Defaults to None.
    """

    def __init__(self, source: str | None = None):
        self.errors: list[GCNParseError] = []
        self.source = source

    def add(self, error: GCNParseError) -> None:
        """
        Append an error or warning to the collection.

        Parameters
        ----------
        error : GCNParseError
            The error to record.
        """
        self.errors.append(error)

    def raise_if_errors(self) -> None:
        """Raise every collected issue as one :class:`GCNErrorCollection`, or return when nothing was collected."""
        if self.errors:
            raise GCNErrorCollection(self.errors, self.source)

    def raise_first(self) -> None:
        """Raise the first collected issue with error severity, or return when there is none."""
        for error in self.errors:
            if error.severity == Severity.ERROR:
                raise error

    @property
    def has_errors(self) -> bool:
        """True if at least one collected issue has error severity. Warnings alone give False."""
        return any(error.severity == Severity.ERROR for error in self.errors)

    @property
    def warnings(self) -> list[GCNParseError]:
        """The collected issues with warning severity."""
        return [error for error in self.errors if error.severity == Severity.WARNING]

    def __len__(self) -> int:
        return len(self.errors)

    def __bool__(self) -> bool:
        return len(self.errors) > 0

    def __iter__(self):
        return iter(self.errors)
