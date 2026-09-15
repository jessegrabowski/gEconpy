import sys

import pytest

from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import (
    GCNErrorCollection,
    GCNGrammarError,
    GCNParseError,
    GCNSemanticError,
    ParseLocation,
    Severity,
)
from gEconpy.parser.formatting import Colors, ErrorFormatter


@pytest.fixture
def formatter():
    return ErrorFormatter(use_color=False)


class TestFormatError:
    def test_layout_of_full_error(self, formatter):
        source = "block TEST\n{\n    identities { Y[] = C[] + ; };\n};"
        err = GCNGrammarError(
            "Expected expression after operator",
            location=ParseLocation(3, 30, filename="model.gcn"),
            code=ErrorCode.E006,
            annotation="missing operand",
            suggestions=["I[]"],
            notes=["Every operator needs two operands"],
        )
        assert formatter.format_error(err, source) == "\n".join(
            [
                "error[E006]: Expected expression after operator",
                "  --> model.gcn:3:30",
                "     |",
                "   1 | block TEST",
                "   2 | {",
                "   3 |     identities { Y[] = C[] + ; };",
                "     |                              ^ missing operand",
                "   4 | };",
                "     |",
                "   = help: Did you mean 'I[]'?",
                "   = note: Every operator needs two operands",
            ]
        )

    def test_location_defaults_to_input_placeholder(self, formatter):
        err = GCNSemanticError("Unknown distribution", symbol_name="Betta", location=ParseLocation(1, 36))
        output = formatter.format_error(err, "block TEST { calibration { alpha ~ Betta(a=1, b=1); }; };")
        assert "  --> <input>:1:36" in output

    def test_multiple_suggestions(self, formatter):
        err = GCNSemanticError(
            "Unknown block component",
            symbol_name="defintons",
            location=ParseLocation(1, 14),
            suggestions=["definitions", "identities"],
        )
        output = formatter.format_error(err, "block TEST { defintons { u[] = log(C[]); }; };")
        assert "= help: Did you mean one of: definitions, identities?" in output

    def test_context_lines_are_clipped_to_source(self, formatter):
        source = "block A { };\nblock A { };"
        err = GCNSemanticError("Duplicate block name", symbol_name="A", location=ParseLocation(2, 7))
        assert formatter.format_error(err, source) == "\n".join(
            [
                "error: Duplicate block name: 'A'",
                "  --> <input>:2:7",
                "     |",
                "   1 | block A { };",
                "   2 | block A { };",
                "     |       ^",
                "     |",
            ]
        )

    def test_custom_context_lines(self):
        formatter = ErrorFormatter(use_color=False, context_lines=1)
        source = "line1\nline2\nY[] = ;\nline4\nline5"
        err = GCNGrammarError("Empty right-hand side", location=ParseLocation(3, 7))
        output = formatter.format_error(err, source)
        assert "line2" in output
        assert "line4" in output
        assert "line1" not in output
        assert "line5" not in output

    def test_no_location_omits_excerpt(self, formatter):
        output = formatter.format_error(GCNSemanticError("Model contains no equations"), None)
        assert output == "error: Model contains no equations"

    def test_warning_severity_label(self, formatter):
        err = GCNParseError(
            "Parameter 'orphan_param' calibrated but never used in equations",
            severity=Severity.WARNING,
            location=ParseLocation(1, 28),
        )
        output = formatter.format_error(err, "block TEST { calibration { orphan_param = 1.0; }; };")
        assert output.startswith("warning: Parameter")

    def test_pointer_uses_span_length(self, formatter):
        err = GCNSemanticError(
            "Unknown block component",
            symbol_name="constrains",
            location=ParseLocation(1, 14, end_line=1, end_column=24),
        )
        output = formatter.format_error(err, "block TEST { constrains { Y[] = C[] : lambda[]; }; };")
        assert "     |              ^^^^^^^^^^" in output

    def test_pointer_single_char_without_span(self, formatter):
        err = GCNGrammarError("Expected expression", location=ParseLocation(1, 33))
        output = formatter.format_error(err, "block TEST { identities { Y[] = ; }; };")
        pointer_lines = [line for line in output.split("\n") if "^" in line]
        assert len(pointer_lines) == 1
        assert pointer_lines[0].count("^") == 1

    def test_location_beyond_source_falls_back_to_stored_line(self, formatter):
        err = GCNGrammarError(
            "Unexpected end of file",
            location=ParseLocation(10, 1, source_line="# truncated content"),
        )
        output = formatter.format_error(err, "block TEST { };")
        assert "  10 | # truncated content" in output


class TestFormatErrorCollection:
    def test_single_error(self, formatter):
        source = "block TEST { calibration { alpha = ; }; };"
        collection = GCNErrorCollection(
            [GCNGrammarError("Missing value after '='", location=ParseLocation(1, 36))], source
        )
        output = formatter.format_error_collection(collection)
        assert "Missing value" in output
        assert output.endswith("error: aborting due to 1 previous error")

    def test_multiple_errors_separated_by_blank_line(self, formatter):
        source = "block A { };\nblock A { };"
        errors = [
            GCNSemanticError("Duplicate block name", symbol_name="A", location=ParseLocation(2, 7)),
            GCNGrammarError("Block 'A' has no equations", location=ParseLocation(1, 1)),
        ]
        output = formatter.format_error_collection(GCNErrorCollection(errors, source))
        first, second, summary = output.split("\n\n")
        assert first.startswith("error: Duplicate block name")
        assert second.startswith("error[E000]: Block 'A' has no equations")
        assert summary == "error: aborting due to 2 previous errors"

    def test_empty_collection(self, formatter):
        assert formatter.format_error_collection(GCNErrorCollection([])) == ""


class TestColorSupport:
    def test_color_disabled_emits_no_ansi(self, formatter):
        err = GCNGrammarError("Unbalanced braces", code=ErrorCode.E002)
        assert "\x1b[" not in formatter.format_error(err, "block TEST {")

    @pytest.mark.parametrize(
        "isatty, env, expect_color",
        [
            (True, {"TERM": "xterm-256color"}, True),
            (False, {"TERM": "xterm-256color"}, False),
            (True, {"TERM": "xterm-256color", "NO_COLOR": "1"}, False),
            (True, {"TERM": "dumb"}, False),
        ],
        ids=["tty", "not_a_tty", "no_color_env", "dumb_terminal"],
    )
    def test_color_requested_only_emitted_on_capable_terminal(self, monkeypatch, isatty, env, expect_color):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: isatty)
        monkeypatch.delenv("NO_COLOR", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        formatter = ErrorFormatter(use_color=True)
        output = formatter.format_error(GCNGrammarError("Unbalanced braces", code=ErrorCode.E002))

        assert formatter.use_color is expect_color
        assert output == (
            f"{Colors.BOLD_RED}error[E002]{Colors.RESET}: Unbalanced braces"
            if expect_color
            else "error[E002]: Unbalanced braces"
        )

    def test_colors_class_has_expected_codes(self):
        assert Colors.RED.startswith("\x1b[")
        assert Colors.RESET == "\x1b[0m"
        assert Colors.BOLD_RED.startswith("\x1b[")
