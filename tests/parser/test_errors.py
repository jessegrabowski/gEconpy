import pytest

from gEconpy.parser.error_catalog import ErrorCode
from gEconpy.parser.errors import (
    ErrorCollector,
    GCNErrorCollection,
    GCNGrammarError,
    GCNParseError,
    GCNSemanticError,
    ParseLocation,
    Severity,
)


class TestParseLocation:
    def test_basic_creation(self):
        loc = ParseLocation(line=5, column=10)
        assert loc.line == 5
        assert loc.column == 10
        assert loc.source_line == ""
        assert loc.filename == ""

    def test_full_creation(self):
        loc = ParseLocation(
            line=10,
            column=15,
            source_line="    Y[] = C[] + I[];",
            filename="model.gcn",
        )
        assert loc.line == 10
        assert loc.column == 15
        assert loc.source_line == "    Y[] = C[] + I[];"
        assert loc.filename == "model.gcn"

    def test_format_location_with_filename(self):
        loc = ParseLocation(line=10, column=5, filename="model.gcn")
        assert loc.format_location() == "model.gcn:10:5"

    def test_format_location_without_filename(self):
        loc = ParseLocation(line=10, column=5)
        assert loc.format_location() == "line 10, column 5"

    def test_format_pointer(self):
        loc = ParseLocation(line=1, column=5, source_line="    X[] = Y[];")
        result = loc.format_pointer()
        lines = result.split("\n")
        assert lines[0] == "    X[] = Y[];"
        assert lines[1] == "    ^"

    def test_format_pointer_at_start(self):
        loc = ParseLocation(line=1, column=1, source_line="X[] = Y[];")
        result = loc.format_pointer()
        lines = result.split("\n")
        assert lines[0] == "X[] = Y[];"
        assert lines[1] == "^"

    def test_format_pointer_empty_source(self):
        loc = ParseLocation(line=1, column=5)
        assert loc.format_pointer() == ""

    def test_format_pointer_custom_char(self):
        loc = ParseLocation(line=1, column=3, source_line="abc")
        result = loc.format_pointer(pointer_char="~")
        assert "~" in result
        assert "^" not in result

    def test_str_representation(self):
        loc = ParseLocation(line=5, column=10, filename="test.gcn")
        assert str(loc) == "test.gcn:5:10"

    def test_frozen(self):
        loc = ParseLocation(line=5, column=10)
        with pytest.raises(AttributeError):
            loc.line = 20


class TestGCNParseError:
    def test_simple_message(self):
        err = GCNParseError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"
        assert err.location is None
        assert err.suggestions == []

    def test_message_with_location(self):
        loc = ParseLocation(line=10, column=5, source_line="    bad code;")
        err = GCNParseError("Invalid syntax", location=loc)
        msg = str(err)
        assert "Invalid syntax" in msg
        assert "line 10, column 5" in msg
        assert "bad code" in msg
        assert "^" in msg

    def test_message_with_single_suggestion(self):
        err = GCNParseError("Unknown variable 'Consumptin'", suggestions=["Consumption"])
        msg = str(err)
        assert "Did you mean: Consumption?" in msg

    def test_message_with_multiple_suggestions(self):
        err = GCNParseError("Unknown variable", suggestions=["C", "Consumption", "Capital"])
        msg = str(err)
        assert "Did you mean one of:" in msg
        assert "C" in msg
        assert "Consumption" in msg
        assert "Capital" in msg

    def test_message_with_context(self):
        err = GCNParseError("Missing semicolon", context="block HOUSEHOLD")
        msg = str(err)
        assert "Missing semicolon" in msg
        assert "in block HOUSEHOLD" in msg

    def test_full_error_formatting(self):
        loc = ParseLocation(
            line=15,
            column=10,
            source_line="    alpha ~ Beta(mena=0.5);",
            filename="rbc.gcn",
        )
        err = GCNParseError(
            message="Unknown parameter 'mena'",
            location=loc,
            suggestions=["mean"],
            context="distribution for alpha",
        )
        msg = str(err)
        assert "Unknown parameter 'mena'" in msg
        assert "distribution for alpha" in msg
        assert "rbc.gcn:15:10" in msg
        assert "mena" in msg
        assert "Did you mean: mean?" in msg

    def test_with_location(self):
        err = GCNParseError("Something wrong")
        loc = ParseLocation(line=5, column=3)
        new_err = err.with_location(loc)

        assert err.location is None
        assert new_err.location == loc
        assert new_err.message == err.message

    def test_with_context(self):
        err = GCNParseError("Something wrong")
        new_err = err.with_context("block FIRM")

        assert err.context == ""
        assert new_err.context == "block FIRM"
        assert new_err.message == err.message


class TestGCNGrammarError:
    def test_basic_grammar_error(self):
        err = GCNGrammarError("Unexpected token")
        assert "Unexpected token" in str(err)

    def test_grammar_error_with_expected_and_found(self):
        err = GCNGrammarError("Syntax error", expected=";", found="}")
        msg = str(err)
        assert "Expected ';'" in msg
        assert "found '}'" in msg

    def test_grammar_error_with_multiple_expected(self):
        err = GCNGrammarError("Unexpected token", expected=[";", "}", "+"], found="@")
        msg = str(err)
        assert "Expected one of" in msg
        assert "';'" in msg
        assert "'}'" in msg
        assert "'+'" in msg
        assert "found '@'" in msg

    def test_grammar_error_expected_only(self):
        err = GCNGrammarError("Missing token", expected=";")
        msg = str(err)
        assert "Expected ';'" in msg

    def test_grammar_error_with_context(self):
        err = GCNGrammarError("Missing semicolon", expected=";", context="block HOUSEHOLD")
        msg = str(err)
        assert "in block HOUSEHOLD" in msg


class TestGCNSemanticError:
    def test_basic_semantic_error(self):
        err = GCNSemanticError("Undefined variable")
        assert "Undefined variable" in str(err)

    def test_semantic_error_with_symbol(self):
        err = GCNSemanticError("Undefined variable", symbol_name="Consumptin")
        msg = str(err)
        assert "Undefined variable" in msg
        assert "'Consumptin'" in msg

    def test_semantic_error_with_suggestions(self):
        err = GCNSemanticError(
            "Undefined variable",
            symbol_name="Consumptin",
            suggestions=["Consumption", "C"],
        )
        msg = str(err)
        assert "Did you mean one of:" in msg
        assert "Consumption" in msg


class TestGCNParseErrorLSP:
    def test_to_lsp_diagnostic_basic(self):
        err = GCNParseError("Test error")
        diag = err.to_lsp_diagnostic()
        assert diag["message"] == "Test error"
        assert diag["severity"] == 1
        assert diag["source"] == "gEconpy"

    def test_to_lsp_diagnostic_with_location(self):
        loc = ParseLocation(line=5, column=10, end_line=5, end_column=15)
        err = GCNParseError("Test error", location=loc)
        diag = err.to_lsp_diagnostic()
        assert diag["range"]["start"]["line"] == 4
        assert diag["range"]["start"]["character"] == 9
        assert diag["range"]["end"]["line"] == 4
        assert diag["range"]["end"]["character"] == 14

    def test_to_lsp_diagnostic_with_code(self):
        err = GCNParseError("Test error", code=ErrorCode.E001)
        diag = err.to_lsp_diagnostic()
        assert diag["code"] == "E001"

    def test_to_lsp_diagnostic_warning_severity(self):
        err = GCNParseError("Warning", severity=Severity.WARNING)
        diag = err.to_lsp_diagnostic()
        assert diag["severity"] == 2


class TestGCNErrorCollection:
    def test_empty_collection(self):
        exc = GCNErrorCollection([])
        assert len(exc) == 0
        assert not exc.has_errors

    def test_single_error(self):
        errors = [GCNSemanticError("Error 1")]
        exc = GCNErrorCollection(errors)
        assert len(exc) == 1
        assert exc.has_errors

    def test_multiple_errors(self):
        errors = [
            GCNSemanticError("Error 1", location=ParseLocation(1, 5)),
            GCNSemanticError("Error 2", location=ParseLocation(3, 10)),
        ]
        exc = GCNErrorCollection(errors)
        assert len(exc) == 2

    def test_iteration(self):
        errors = [GCNSemanticError("Error 1"), GCNSemanticError("Error 2")]
        exc = GCNErrorCollection(errors)
        assert len(list(exc)) == 2


class TestErrorCollector:
    def test_empty_collector(self):
        collector = ErrorCollector()
        assert len(collector) == 0
        assert not collector

    def test_add_error(self):
        collector = ErrorCollector()
        collector.add(GCNSemanticError("Error 1"))
        assert len(collector) == 1

    def test_raise_if_errors_raises_collection(self):
        collector = ErrorCollector()
        collector.add(GCNSemanticError("Error 1"))
        with pytest.raises(GCNErrorCollection):
            collector.raise_if_errors()

    def test_raise_if_errors_noop_when_empty(self):

        collector = ErrorCollector()
        collector.raise_if_errors()
