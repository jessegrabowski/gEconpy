import os
import sys

from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import GCNErrorCollection, GCNParseError, ParseLocation, Severity


class Colors:
    """ANSI escape codes for terminal output."""

    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"

    RED = "\x1b[31m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    CYAN = "\x1b[36m"

    BOLD_RED = "\x1b[1;31m"
    BOLD_YELLOW = "\x1b[1;33m"
    BOLD_BLUE = "\x1b[1;34m"


class ErrorFormatter:
    """
    Render parse errors for the terminal in the style of a compiler diagnostic, with a source excerpt and a caret.

    Parameters
    ----------
    use_color : bool, optional
        Emit ANSI color codes. Colors are only emitted when the terminal also supports them. Defaults to True.
    context_lines : int, optional
        Number of source lines to show before and after the error line. Defaults to 2.
    """

    def __init__(self, use_color: bool = True, context_lines: int = 2):
        self.use_color = use_color and _supports_color()
        self.context_lines = context_lines

    def format_error(self, error: GCNParseError, source: str | None = None) -> str:
        """
        Render a single error with its source excerpt, suggestions, and notes.

        Parameters
        ----------
        error : GCNParseError
            The error to render.
        source : str, optional
            The full source text, used to show lines around the error. Without it, only the ``source_line`` stored
            on the error's location is shown. Defaults to None.

        Returns
        -------
        output : str
            The rendered error.
        """
        severity_label = self._format_severity(error.severity, error.code)
        parts = [f"{severity_label}: {error.message}"]

        if error.location:
            filename = error.location.filename or "<input>"
            parts.append(self._format_location(filename, error.location.line, error.location.column))
            parts.append(self._color("     |", Colors.BLUE))
            parts.extend(self._format_source_excerpt(error.location, source, error.annotation))
            parts.append(self._color("     |", Colors.BLUE))

        if error.suggestions:
            parts.append(self._format_help(error.suggestions))

        parts.extend(self._format_note(note) for note in error.notes)

        return "\n".join(parts)

    def format_error_collection(self, collection: GCNErrorCollection) -> str:
        """
        Render every error in a collection, separated by blank lines and followed by a count.

        Parameters
        ----------
        collection : GCNErrorCollection
            The errors to render. Its ``source`` supplies the excerpt context.

        Returns
        -------
        output : str
            The rendered errors, or an empty string for an empty collection.
        """
        if not collection.errors:
            return ""

        rendered = [self.format_error(error, collection.source) for error in collection.errors]

        error_count = len(collection.errors)
        noun = "error" if error_count == 1 else "errors"
        summary = self._color(f"error: aborting due to {error_count} previous {noun}", Colors.BOLD_RED)

        return "\n\n".join(rendered) + f"\n\n{summary}"

    def _format_source_excerpt(self, location: ParseLocation, source: str | None, annotation: str) -> list[str]:
        source_lines = source.split("\n") if source else []

        if source_lines and location.line <= len(source_lines):
            first_line = max(1, location.line - self.context_lines)
            last_line = min(len(source_lines), location.line + self.context_lines)

            excerpt = []
            for line_number in range(first_line, last_line + 1):
                is_error_line = line_number == location.line
                excerpt.append(self._format_source_line(line_number, source_lines[line_number - 1], is_error_line))
                if is_error_line:
                    excerpt.append(self._format_pointer(location, annotation))
            return excerpt

        if location.source_line:
            return [
                self._format_source_line(location.line, location.source_line, is_error_line=True),
                self._format_pointer(location, annotation),
            ]

        return []

    def _color(self, text: str, color: str) -> str:
        if self.use_color:
            return f"{color}{text}{Colors.RESET}"
        return text

    def _format_severity(self, severity: Severity, code: ErrorCode | None) -> str:
        label = f"{severity.value}[{code.name}]" if code else severity.value

        if severity == Severity.ERROR:
            return self._color(label, Colors.BOLD_RED)
        if severity == Severity.WARNING:
            return self._color(label, Colors.BOLD_YELLOW)
        return self._color(label, Colors.BOLD)

    def _format_location(self, filename: str, line: int, column: int) -> str:
        location = f"{filename}:{line}:{column}"
        return f"  --> {self._color(location, Colors.BLUE)}"

    def _format_source_line(self, line_number: int, line_text: str, is_error_line: bool = False) -> str:
        gutter_color = Colors.BOLD_BLUE if is_error_line else Colors.BLUE
        gutter = self._color(f"{line_number:4}", gutter_color)

        line_text = line_text.rstrip()
        if line_text:
            return f"{gutter} | {line_text}"
        return f"{gutter} |"

    def _format_pointer(self, location: ParseLocation, annotation: str = "") -> str:
        padding = " " * (location.column - 1)
        pointer = self._color("^" * location.pointer_length, Colors.RED)
        gutter = self._color("     | ", Colors.BLUE)

        if annotation:
            return f"{gutter}{padding}{pointer} {self._color(annotation, Colors.RED)}"
        return f"{gutter}{padding}{pointer}"

    def _format_help(self, suggestions: list[str]) -> str:
        help_label = self._color("= help:", Colors.CYAN)
        if len(suggestions) == 1:
            return f"   {help_label} Did you mean '{suggestions[0]}'?"
        return f"   {help_label} Did you mean one of: {', '.join(suggestions)}?"

    def _format_note(self, note: str) -> str:
        note_label = self._color("= note:", Colors.CYAN)
        return f"   {note_label} {note}"


def _supports_color() -> bool:
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return os.environ.get("TERM") != "dumb"
