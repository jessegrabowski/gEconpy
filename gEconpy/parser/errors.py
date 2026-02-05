from dataclasses import dataclass, field


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
    source_line: str = ""
    filename: str = ""

    def format_pointer(self, pointer_char: str = "^") -> str:
        """
        Format the source line with a pointer to the error column.

        Parameters
        ----------
        pointer_char : str, optional
            Character to use for the pointer. Default is "^".

        Returns
        -------
        str
            The source line followed by a pointer line.
        """
        if not self.source_line:
            return ""

        # Adjust column to be 0-based for string indexing
        col_index = max(0, self.column - 1)
        pointer_line = " " * col_index + pointer_char
        return f"{self.source_line}\n{pointer_line}"

    def format_location(self) -> str:
        """
        Format the location as a string.

        Returns
        -------
        str
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
        Suggested fixes or alternatives.
    context : str, optional
        Additional context about the error (e.g., block name).

    Attributes
    ----------
    message : str
        The main error message.
    location : ParseLocation or None
        The source location if available.
    suggestions : list of str
        List of suggested fixes.
    context : str
        Additional context string.

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
    ):
        self.message = message
        self.location = location
        self.suggestions = suggestions or []
        self.context = context
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the complete error message with location and suggestions."""
        parts = [self.message]

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
        GCNParseError
            A new error instance with the location set.
        """
        return self.__class__(
            message=self.message,
            location=location,
            suggestions=self.suggestions,
            context=self.context,
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
        GCNParseError
            A new error instance with the context set.
        """
        return self.__class__(
            message=self.message,
            location=self.location,
            suggestions=self.suggestions,
            context=context,
        )


class GCNLexerError(GCNParseError):
    """
    Error during tokenization/lexing of GCN source.

    Raised when the lexer encounters characters or sequences it cannot
    recognize as valid tokens.

    Parameters
    ----------
    message : str
        Description of what went wrong during tokenization.
    invalid_text : str, optional
        The text that could not be tokenized.
    location : ParseLocation, optional
        Where in the source the error occurred.

    Examples
    --------
    .. testcode::

        from gEconpy.parser.errors import GCNLexerError, ParseLocation

        err = GCNLexerError(
            "Unexpected character",
            invalid_text="@",
            location=ParseLocation(line=3, column=15)
        )
    """

    def __init__(
        self,
        message: str,
        invalid_text: str = "",
        location: ParseLocation | None = None,
        suggestions: list[str] | None = None,
    ):
        self.invalid_text = invalid_text
        if invalid_text and invalid_text not in message:
            message = f"{message}: '{invalid_text}'"
        super().__init__(message=message, location=location, suggestions=suggestions)


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

        super().__init__(message=message, location=location, context=context)


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
    ):
        self.symbol_name = symbol_name
        if symbol_name and symbol_name not in message:
            message = f"{message}: '{symbol_name}'"
        super().__init__(
            message=message,
            location=location,
            suggestions=suggestions,
            context=context,
        )


@dataclass
class ValidationError:
    """
    A single validation issue found during AST validation.

    This is a lightweight container used during validation passes
    before being converted to a full GCNSemanticError if needed.

    Attributes
    ----------
    message : str
        Description of the issue.
    severity : str
        One of 'error', 'warning', 'info'.
    location : ParseLocation, optional
        Where in the source the issue was found.
    suggestions : list of str
        Suggested fixes.

    Examples
    --------
    .. testcode::

        from gEconpy.parser.errors import ValidationError, ParseLocation

        issue = ValidationError(
            message="Variable 'X' used but never defined",
            severity="error",
            location=ParseLocation(line=10, column=5)
        )
        if issue.is_error:
            raise issue.to_exception()

    .. testoutput::

        Traceback (most recent call last):
            ...
        gEconpy.parser.errors.GCNSemanticError: Variable 'X' used but never defined
          at line 10, column 5
    """

    message: str
    severity: str = "error"  # 'error', 'warning', 'info'
    location: ParseLocation | None = None
    suggestions: list[str] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        """Return True if this is an error (not just a warning)."""
        return self.severity == "error"

    @property
    def is_warning(self) -> bool:
        """Return True if this is a warning."""
        return self.severity == "warning"

    def to_exception(self) -> GCNSemanticError:
        """
        Convert this validation error to an exception.

        Returns
        -------
        GCNSemanticError
            An exception that can be raised.
        """
        return GCNSemanticError(
            message=self.message,
            location=self.location,
            suggestions=self.suggestions,
        )

    def __str__(self) -> str:
        prefix = f"[{self.severity.upper()}]"
        if self.location:
            return f"{prefix} {self.location}: {self.message}"
        return f"{prefix} {self.message}"


class ValidationErrorCollection:
    """
    A collection of validation errors and warnings.

    This class accumulates validation issues during an AST validation pass
    and provides methods to check if there are errors and raise them.

    Examples
    --------
    .. testcode::

        from gEconpy.parser.errors import ValidationErrorCollection, ParseLocation

        loc = ParseLocation(line=10, column=5)
        errors = ValidationErrorCollection()
        errors.add_error("Undefined variable 'X'", location=loc)
        errors.add_warning("Variable 'Y' defined but never used")
        print(errors.has_errors)
        print(len(errors.warnings))

    .. testoutput::

        True
        1
    """

    def __init__(self):
        self._errors: list[ValidationError] = []

    def add(self, error: ValidationError) -> None:
        """Add a validation error to the collection."""
        self._errors.append(error)

    def add_error(
        self,
        message: str,
        location: ParseLocation | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        """Add an error-level issue."""
        self.add(
            ValidationError(
                message=message,
                severity="error",
                location=location,
                suggestions=suggestions or [],
            )
        )

    def add_warning(
        self,
        message: str,
        location: ParseLocation | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        """Add a warning-level issue."""
        self.add(
            ValidationError(
                message=message,
                severity="warning",
                location=location,
                suggestions=suggestions or [],
            )
        )

    @property
    def has_errors(self) -> bool:
        """Return True if there are any error-level issues."""
        return any(e.is_error for e in self._errors)

    @property
    def errors(self) -> list[ValidationError]:
        """Return all error-level issues."""
        return [e for e in self._errors if e.is_error]

    @property
    def warnings(self) -> list[ValidationError]:
        """Return all warning-level issues."""
        return [e for e in self._errors if e.is_warning]

    @property
    def all_issues(self) -> list[ValidationError]:
        """Return all issues."""
        return list(self._errors)

    def raise_first(self) -> None:
        """Raise the first error as an exception."""
        for error in self._errors:
            if error.is_error:
                raise error.to_exception()

    def raise_all(self) -> None:
        """
        Raise an exception with all errors combined.

        Raises
        ------
        GCNSemanticError
            An exception containing all error messages.
        """
        error_list = self.errors
        if not error_list:
            return

        if len(error_list) == 1:
            raise error_list[0].to_exception()

        combined_message = f"Found {len(error_list)} errors:\n"
        combined_message += "\n".join(f"  {i + 1}. {e.message}" for i, e in enumerate(error_list))
        raise GCNSemanticError(combined_message)

    def __len__(self) -> int:
        return len(self._errors)

    def __bool__(self) -> bool:
        return len(self._errors) > 0

    def __iter__(self):
        return iter(self._errors)
