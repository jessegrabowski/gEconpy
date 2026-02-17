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


class TestErrorFormatter:
    @pytest.fixture
    def formatter(self):
        return ErrorFormatter(use_color=False)

    def test_format_error_includes_message(self, formatter):
        source = "block TEST { calibration { alpha = 0.5; alpha = 0.6; }; };"
        err = GCNSemanticError(
            "Duplicate parameter definition",
            symbol_name="alpha",
            location=ParseLocation(1, 41),
        )
        output = formatter.format_error(err, source)
        assert "Duplicate parameter" in output

    def test_format_error_includes_code(self, formatter):
        source = "block TEST { identites { Y[] = C[]; }; };"
        err = GCNGrammarError(
            "Unknown block component",
            expected="identities",
            found="identites",
            location=ParseLocation(1, 14),
            code=ErrorCode.E003,
        )
        output = formatter.format_error(err, source)
        assert "E003" in output

    def test_format_error_includes_location(self, formatter):
        source = "block TEST { calibration { alpha ~ Betta(a=1, b=1); }; };"
        err = GCNSemanticError(
            "Unknown distribution",
            symbol_name="Betta",
            location=ParseLocation(1, 36),
        )
        output = formatter.format_error(err, source)
        assert ":1:" in output
        assert ":36" in output

    def test_format_error_includes_source_snippet(self, formatter):
        source = "block TEST\n{\n    identities { Y[] = C[] + ; };\n};"
        err = GCNGrammarError(
            "Expected expression after operator",
            location=ParseLocation(3, 42),
        )
        output = formatter.format_error(err, source)
        assert "Y[] = C[] +" in output

    def test_format_error_includes_pointer(self, formatter):
        source = "block TEST { identities { Y[] = }; };"
        err = GCNGrammarError(
            "Missing right-hand side",
            location=ParseLocation(1, 33),
        )
        output = formatter.format_error(err, source)
        assert "^" in output

    def test_format_error_includes_suggestions(self, formatter):
        source = "block TEST { calibration { rho ~ Nomal(mu=0, sigma=1); }; };"
        err = GCNSemanticError(
            "Unknown distribution",
            symbol_name="Nomal",
            location=ParseLocation(1, 34),
            suggestions=["Normal"],
        )
        output = formatter.format_error(err, source)
        assert "Normal" in output
        assert "Did you mean" in output

    def test_format_error_multiple_suggestions(self, formatter):
        source = "block TEST { defintons { u[] = log(C[]); }; };"
        err = GCNSemanticError(
            "Unknown block component",
            symbol_name="defintons",
            location=ParseLocation(1, 14),
            suggestions=["definitions", "identities"],
        )
        output = formatter.format_error(err, source)
        assert "definitions" in output
        assert "identities" in output
        assert "one of" in output

    def test_format_error_shows_context_lines(self, formatter):
        source = "block A { };\nblock B { };\nblock A { };\nblock C { };\nblock D { };"
        err = GCNSemanticError(
            "Duplicate block name",
            symbol_name="A",
            location=ParseLocation(3, 7),
        )
        output = formatter.format_error(err, source)
        assert "block A" in output
        assert "block B" in output
        assert "block C" in output

    def test_format_error_no_location(self, formatter):
        err = GCNSemanticError("Model contains no equations")
        output = formatter.format_error(err, None)
        assert "Model contains no equations" in output
        assert "-->" not in output

    def test_format_error_warning_severity(self, formatter):
        source = "block TEST { calibration { orphan_param = 1.0; }; };"
        err = GCNParseError(
            "Parameter 'orphan_param' calibrated but never used in equations",
            severity=Severity.WARNING,
            location=ParseLocation(1, 28),
        )
        output = formatter.format_error(err, source)
        assert "warning" in output

    def test_format_error_with_special_characters(self, formatter):
        source = "block TEST { identities { Y[] = C[] + I[]; }; };"
        err = GCNGrammarError(
            "Unexpected token",
            found="+",
            location=ParseLocation(1, 37),
        )
        output = formatter.format_error(err, source)
        assert "+" in output


class TestErrorFormatterCollection:
    @pytest.fixture
    def formatter(self):
        return ErrorFormatter(use_color=False)

    def test_format_collection_single_error(self, formatter):
        source = "block TEST { calibration { alpha = ; }; };"
        errors = [GCNGrammarError("Missing value after '='", location=ParseLocation(1, 36))]
        collection = GCNErrorCollection(errors, source)
        output = formatter.format_error_collection(collection)
        assert "Missing value" in output
        assert "1 previous error" in output

    def test_format_collection_multiple_errors(self, formatter):
        source = "block A { };\nblock A { };"
        errors = [
            GCNSemanticError("Duplicate block name", symbol_name="A", location=ParseLocation(2, 7)),
            GCNGrammarError("Block 'A' has no equations", location=ParseLocation(1, 1)),
        ]
        collection = GCNErrorCollection(errors, source)
        output = formatter.format_error_collection(collection)
        assert "Duplicate block" in output
        assert "no equations" in output
        assert "2 previous errors" in output

    def test_format_collection_empty(self, formatter):
        collection = GCNErrorCollection([])
        output = formatter.format_error_collection(collection)
        assert output == ""


class TestColorSupport:
    def test_color_disabled_no_ansi(self):
        formatter = ErrorFormatter(use_color=False)
        err = GCNGrammarError("Unbalanced braces", code=ErrorCode.E002)
        output = formatter.format_error(err, "block TEST {")
        assert "\x1b[" not in output

    def test_colors_class_has_expected_codes(self):
        assert Colors.RED.startswith("\x1b[")
        assert Colors.RESET == "\x1b[0m"
        assert Colors.BOLD_RED.startswith("\x1b[")


class TestFormatterWithSpan:
    @pytest.fixture
    def formatter(self):
        return ErrorFormatter(use_color=False)

    def test_pointer_uses_span_length(self, formatter):
        source = "block TEST { constrains { Y[] = C[] : lambda[]; }; };"
        err = GCNSemanticError(
            "Unknown block component",
            symbol_name="constrains",
            location=ParseLocation(1, 14, end_line=1, end_column=24),
            suggestions=["constraints"],
        )
        output = formatter.format_error(err, source)
        assert "^^^^^^^^^^" in output

    def test_pointer_single_char_without_span(self, formatter):
        source = "block TEST { identities { Y[] = ; }; };"
        err = GCNGrammarError(
            "Expected expression",
            location=ParseLocation(1, 33),
        )
        output = formatter.format_error(err, source)
        lines = output.split("\n")
        pointer_lines = [line for line in lines if "^" in line and "|" in line]
        assert len(pointer_lines) == 1
        assert pointer_lines[0].count("^") == 1


class TestFormatterEdgeCases:
    @pytest.fixture
    def formatter(self):
        return ErrorFormatter(use_color=False)

    def test_error_at_first_line(self, formatter):
        source = "blok TEST { };\nblock B { };"
        err = GCNGrammarError("Expected 'block' keyword", found="blok", location=ParseLocation(1, 1))
        output = formatter.format_error(err, source)
        assert "blok" in output

    def test_error_at_last_line(self, formatter):
        source = "block A { };\nblock B { };\nblock C {"
        err = GCNGrammarError("Unclosed brace - expected '}'", location=ParseLocation(3, 10))
        output = formatter.format_error(err, source)
        assert "block C" in output

    def test_error_beyond_source_lines(self, formatter):
        source = "block TEST { };"
        err = GCNGrammarError(
            "Unexpected end of file",
            location=ParseLocation(10, 1, source_line="# truncated content"),
        )
        output = formatter.format_error(err, source)
        assert "truncated content" in output

    def test_custom_context_lines(self):
        formatter = ErrorFormatter(use_color=False, context_lines=1)
        source = "line1\nline2\nY[] = ;\nline4\nline5"
        err = GCNGrammarError("Empty right-hand side", location=ParseLocation(3, 7))
        output = formatter.format_error(err, source)
        assert "line2" in output
        assert "Y[] = ;" in output
        assert "line4" in output
        assert "line1" not in output
        assert "line5" not in output
