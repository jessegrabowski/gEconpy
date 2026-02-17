import os
import sys

from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNErrorCollection, GCNParseError, Severity


def _supports_color() -> bool:
    """Check if the terminal supports ANSI color codes."""
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return os.environ.get("TERM") != "dumb"


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"

    # Foreground colors
    RED = "\x1b[31m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    CYAN = "\x1b[36m"

    # Bold variants
    BOLD_RED = "\x1b[1;31m"
    BOLD_YELLOW = "\x1b[1;33m"
    BOLD_BLUE = "\x1b[1;34m"


class ErrorFormatter:
    """
    Format errors for terminal output with optional color support.

    Parameters
    ----------
    use_color : bool
        Whether to use ANSI color codes. If True, will also check
        if the terminal supports colors.
    context_lines : int
        Number of lines of context to show before and after the error line.
    """

    def __init__(self, use_color: bool = True, context_lines: int = 2):
        self.use_color = use_color and _supports_color()
        self.context_lines = context_lines

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_color:
            return f"{color}{text}{Colors.RESET}"
        return text

    def _format_severity(self, severity: Severity, code: ErrorCode | None) -> str:
        """Format the severity label with optional error code."""
        severity_str = severity.value

        label = f"{severity_str}[{code.name}]" if code else severity_str

        if severity_str == "error":
            return self._color(label, Colors.BOLD_RED)
        if severity_str == "warning":
            return self._color(label, Colors.BOLD_YELLOW)
        return self._color(label, Colors.BOLD)

    def _format_location(self, filename: str, line: int, column: int) -> str:
        """Format the file location."""
        loc = f"{filename}:{line}:{column}"
        return f"  --> {self._color(loc, Colors.BLUE)}"

    def _format_source_line(self, line_num: int, line_text: str, is_error_line: bool = False) -> str:
        """Format a single source line with line number."""
        num_str = f"{line_num:4}"
        num_str = self._color(num_str, Colors.BOLD_BLUE) if is_error_line else self._color(num_str, Colors.BLUE)
        line_text = line_text.rstrip()
        if line_text:
            return f"{num_str} | {line_text}"
        return f"{num_str} |"

    def _format_pointer(self, column: int, length: int = 1, message: str = "") -> str:
        """Format the error pointer (^^^^) under the source line."""
        padding = " " * (column - 1)
        pointer = "^" * max(1, length)
        pointer_str = self._color(pointer, Colors.RED)
        gutter = self._color("     | ", Colors.BLUE)

        if message:
            return f"{gutter}{padding}{pointer_str} {self._color(message, Colors.RED)}"
        return f"{gutter}{padding}{pointer_str}"

    def _format_help(self, suggestions: list[str]) -> str:
        """Format help/suggestion text."""
        if not suggestions:
            return ""

        help_label = self._color("= help:", Colors.CYAN)
        if len(suggestions) == 1:
            return f"   {help_label} Did you mean '{suggestions[0]}'?"
        return f"   {help_label} Did you mean one of: {', '.join(suggestions)}?"

    def _format_note(self, note: str) -> str:
        """Format a single note line."""
        note_label = self._color("= note:", Colors.CYAN)
        return f"   {note_label} {note}"

    def _format_notes(self, notes: list[str]) -> list[str]:
        """Format all note lines."""
        return [self._format_note(note) for note in notes]

    def format_error(self, error: GCNParseError, source: str | None = None) -> str:
        """
        Format a single error with color and context.

        Parameters
        ----------
        error : GCNParseError
            The error to format.
        source : str, optional
            The source text for context. If not provided, uses error's location.

        Returns
        -------
        output : str
            Formatted error message.
        """
        parts = []

        # Severity and message line
        severity = getattr(error, "severity", "error")
        code = getattr(error, "code", "")
        severity_label = self._format_severity(severity, code)
        parts.append(f"{severity_label}: {error.message}")

        # Get annotation for pointer line
        annotation = getattr(error, "annotation", "")

        # Location line
        if error.location:
            filename = error.location.filename or "<input>"
            parts.append(self._format_location(filename, error.location.line, error.location.column))

            # Source context
            parts.append(self._color("     |", Colors.BLUE))

            # Get context lines from source string or just the error line
            source_lines = source.split("\n") if source else []
            if source_lines and error.location.line <= len(source_lines):
                start_line = max(1, error.location.line - self.context_lines)
                end_line = min(len(source_lines), error.location.line + self.context_lines)

                for line_num in range(start_line, end_line + 1):
                    line_text = source_lines[line_num - 1] if line_num <= len(source_lines) else ""
                    is_error = line_num == error.location.line
                    parts.append(self._format_source_line(line_num, line_text, is_error))

                    # Add pointer after error line
                    if is_error:
                        # Calculate pointer length from end_column if available
                        length = 1
                        if error.location.end_column and error.location.end_line == error.location.line:
                            length = error.location.end_column - error.location.column
                        parts.append(self._format_pointer(error.location.column, length, annotation))

            elif error.location.source_line:
                # Fall back to source_line from location
                parts.append(self._format_source_line(error.location.line, error.location.source_line, True))
                length = 1
                if error.location.end_column and error.location.end_line == error.location.line:
                    length = error.location.end_column - error.location.column
                parts.append(self._format_pointer(error.location.column, length, annotation))

            parts.append(self._color("     |", Colors.BLUE))

        suggestions = getattr(error, "suggestions", [])
        if suggestions:
            parts.append(self._format_help(suggestions))

        notes = getattr(error, "notes", [])
        if notes:
            parts.extend(self._format_notes(notes))

        return "\n".join(parts)

    def format_error_collection(self, collection: GCNErrorCollection) -> str:
        """
        Format multiple errors with separators.

        Parameters
        ----------
        collection : GCNErrorCollection
            The collection of errors to format.

        Returns
        -------
        output : str
            Formatted error messages.
        """
        if not collection.errors:
            return ""

        parts = []
        source = collection.source

        for i, error in enumerate(collection.errors):
            if i > 0:
                parts.append("")
            parts.append(self.format_error(error, source))

        error_count = len(collection.errors)
        if error_count == 1:
            summary = self._color("error: aborting due to 1 previous error", Colors.BOLD_RED)
        else:
            summary = self._color(f"error: aborting due to {error_count} previous errors", Colors.BOLD_RED)
        parts.append("")
        parts.append(summary)

        return "\n".join(parts)
