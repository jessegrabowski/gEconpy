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
    def test_defaults(self):
        loc = ParseLocation(line=5, column=10)
        assert loc.source_line == ""
        assert loc.filename == ""
        assert loc.end_line is None
        assert loc.end_column is None

    @pytest.mark.parametrize(
        "filename, expected",
        [("model.gcn", "model.gcn:10:5"), ("", "line 10, column 5")],
        ids=["with_filename", "without_filename"],
    )
    def test_format_location(self, filename, expected):
        loc = ParseLocation(line=10, column=5, filename=filename)
        assert loc.format_location() == expected
        assert str(loc) == expected

    @pytest.mark.parametrize(
        "loc, expected",
        [
            (ParseLocation(line=1, column=5, source_line="    X[] = Y[];"), "    X[] = Y[];\n    ^"),
            (ParseLocation(line=1, column=1, source_line="X[] = Y[];"), "X[] = Y[];\n^"),
            (ParseLocation(line=1, column=3, end_line=1, end_column=7, source_line="a bcde f"), "a bcde f\n  ^^^^"),
            (ParseLocation(line=1, column=5), ""),
        ],
        ids=["indented", "at_start", "span", "no_source_line"],
    )
    def test_format_pointer(self, loc, expected):
        assert loc.format_pointer() == expected

    def test_format_pointer_custom_char(self):
        loc = ParseLocation(line=1, column=3, source_line="abc")
        assert loc.format_pointer(pointer_char="~") == "abc\n  ~"

    def test_frozen(self):
        loc = ParseLocation(line=5, column=10)
        with pytest.raises(AttributeError):
            loc.line = 20

    def test_to_lsp_range_is_zero_indexed(self):
        loc = ParseLocation(line=5, column=10, end_line=5, end_column=15)
        assert loc.to_lsp_range() == {
            "start": {"line": 4, "character": 9},
            "end": {"line": 4, "character": 14},
        }

    def test_to_lsp_range_without_span_covers_one_character(self):
        loc = ParseLocation(line=5, column=10)
        assert loc.to_lsp_range()["end"] == {"line": 4, "character": 10}


class TestGCNParseError:
    def test_simple_message(self):
        err = GCNParseError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"
        assert err.location is None
        assert err.suggestions == []

    def test_message_with_location(self):
        loc = ParseLocation(line=10, column=5, source_line="    bad code;")
        msg = str(GCNParseError("Invalid syntax", location=loc))
        assert msg == "Invalid syntax\n  at line 10, column 5\n        bad code;\n        ^"

    @pytest.mark.parametrize(
        "suggestions, expected",
        [
            (["Consumption"], "Did you mean: Consumption?"),
            (["C", "Consumption", "Capital"], "Did you mean one of: C, Consumption, Capital?"),
        ],
        ids=["single", "multiple"],
    )
    def test_message_with_suggestions(self, suggestions, expected):
        assert expected in str(GCNParseError("Unknown variable", suggestions=suggestions))

    def test_message_with_context(self):
        msg = str(GCNParseError("Missing semicolon", context="block HOUSEHOLD"))
        assert msg == "Missing semicolon\n  in block HOUSEHOLD"

    def test_message_with_code(self):
        assert str(GCNParseError("Missing semicolon", code=ErrorCode.E001)) == "[E001] Missing semicolon"

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
        assert "in distribution for alpha" in msg
        assert "at rbc.gcn:15:10" in msg
        assert "Did you mean: mean?" in msg

    @pytest.mark.parametrize(
        "error",
        [
            GCNParseError("Something wrong", notes=["a note"], annotation="here"),
            GCNGrammarError("Syntax error", expected=";", found="}"),
            GCNSemanticError("Undefined variable", symbol_name="Consumptin"),
        ],
        ids=["parse", "grammar", "semantic"],
    )
    def test_with_location_and_context_copy_without_mutating(self, error):
        loc = ParseLocation(line=5, column=3)

        relocated = error.with_location(loc)
        recontextualized = error.with_context("block FIRM")

        assert error.location is None
        assert error.context == ""
        assert type(relocated) is type(error)
        assert relocated.location == loc
        assert relocated.message == error.message
        assert relocated.notes == error.notes
        assert relocated.annotation == error.annotation
        assert recontextualized.context == "block FIRM"
        assert recontextualized.message == error.message


class TestGCNGrammarError:
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, "Unexpected token"),
            ({"expected": ";", "found": "}"}, "Unexpected token. Expected ';', found '}'"),
            ({"expected": [";", "}", "+"], "found": "@"}, "Unexpected token. Expected one of ';', '}', '+', found '@'"),
            ({"expected": ";"}, "Unexpected token. Expected ';'"),
            ({"found": "@"}, "Unexpected token. Found '@'"),
        ],
        ids=["bare", "expected_and_found", "multiple_expected", "expected_only", "found_only"],
    )
    def test_message_names_expected_and_found(self, kwargs, expected):
        error = GCNGrammarError("Unexpected token", **kwargs)
        assert error.message == expected
        assert str(error) == f"[E000] {expected}"

    def test_grammar_error_with_context(self):
        msg = str(GCNGrammarError("Missing semicolon", expected=";", context="block HOUSEHOLD"))
        assert "in block HOUSEHOLD" in msg


class TestGCNSemanticError:
    def test_symbol_name_appended_once(self):
        assert str(GCNSemanticError("Undefined variable", symbol_name="X")) == "Undefined variable: 'X'"
        assert str(GCNSemanticError("Undefined variable 'X'", symbol_name="X")) == "Undefined variable 'X'"

    def test_semantic_error_with_suggestions(self):
        err = GCNSemanticError(
            "Undefined variable",
            symbol_name="Consumptin",
            suggestions=["Consumption", "C"],
        )
        assert "Did you mean one of: Consumption, C?" in str(err)


class TestGCNParseErrorLSP:
    def test_to_lsp_diagnostic_basic(self):
        diag = GCNParseError("Test error").to_lsp_diagnostic()
        assert diag["message"] == "Test error"
        assert diag["severity"] == 1
        assert diag["source"] == "gEconpy"
        assert diag["range"]["start"] == {"line": 0, "character": 0}
        assert "code" not in diag

    def test_to_lsp_diagnostic_with_location(self):
        loc = ParseLocation(line=5, column=10, end_line=5, end_column=15)
        diag = GCNParseError("Test error", location=loc).to_lsp_diagnostic()
        assert diag["range"] == loc.to_lsp_range()

    def test_to_lsp_diagnostic_with_code_and_suggestions(self):
        diag = GCNParseError("Test error", code=ErrorCode.E001, suggestions=["alpha"]).to_lsp_diagnostic()
        assert diag["code"] == "E001"
        assert diag["codeDescription"]["href"].endswith("E001.html")
        assert diag["data"] == {"suggestions": ["alpha"]}

    @pytest.mark.parametrize(
        "severity, lsp_severity",
        [(Severity.ERROR, 1), (Severity.WARNING, 2), (Severity.INFO, 3), (Severity.HINT, 4)],
    )
    def test_to_lsp_diagnostic_severity(self, severity, lsp_severity):
        assert GCNParseError("x", severity=severity).to_lsp_diagnostic()["severity"] == lsp_severity


class TestGCNErrorCollection:
    def test_empty_collection(self):
        exc = GCNErrorCollection([])
        assert len(exc) == 0
        assert not exc.has_errors
        assert str(exc) == "No errors"

    def test_single_error_uses_its_message(self):
        exc = GCNErrorCollection([GCNSemanticError("Error 1")])
        assert len(exc) == 1
        assert exc.has_errors
        assert str(exc) == "Error 1"

    def test_multiple_errors_are_numbered(self):
        errors = [
            GCNSemanticError("Error 1", location=ParseLocation(1, 5)),
            GCNSemanticError("Error 2", location=ParseLocation(3, 10)),
        ]
        exc = GCNErrorCollection(errors)
        assert list(exc) == errors
        assert exc[1] is errors[1]
        assert str(exc).startswith("Found 2 errors:\n[1] Error 1")
        assert "\n[2] Error 2" in str(exc)

    def test_to_lsp_diagnostics(self):
        exc = GCNErrorCollection([GCNSemanticError("Error 1"), GCNSemanticError("Error 2")])
        assert [diag["message"] for diag in exc.to_lsp_diagnostics()] == ["Error 1", "Error 2"]


class TestErrorCollector:
    def test_empty_collector(self):
        collector = ErrorCollector()
        assert len(collector) == 0
        assert not collector
        assert not collector.has_errors
        collector.raise_if_errors()
        collector.raise_first()

    def test_add_error(self):
        collector = ErrorCollector()
        collector.add(GCNSemanticError("Error 1"))
        assert len(collector) == 1
        assert collector

    def test_raise_if_errors_raises_collection_with_source(self):
        collector = ErrorCollector(source="block A { };")
        collector.add(GCNSemanticError("Error 1"))
        with pytest.raises(GCNErrorCollection) as exc_info:
            collector.raise_if_errors()
        assert exc_info.value.source == "block A { };"

    def test_warnings_do_not_count_as_errors(self):
        warning = GCNSemanticError("Unused parameter", severity=Severity.WARNING)
        error = GCNSemanticError("Undefined variable")

        collector = ErrorCollector()
        collector.add(warning)
        assert not collector.has_errors
        assert collector.warnings == [warning]
        collector.raise_first()

        collector.add(error)
        assert collector.has_errors
        with pytest.raises(GCNSemanticError, match="Undefined variable"):
            collector.raise_first()
