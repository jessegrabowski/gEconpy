import pytest

from gEconpy.parser.errors import (
    GCNGrammarError,
    GCNLexerError,
    GCNParseError,
    GCNSemanticError,
    ParseLocation,
    ValidationError,
    ValidationErrorCollection,
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


class TestGCNLexerError:
    def test_basic_lexer_error(self):
        err = GCNLexerError("Unexpected character")
        assert "Unexpected character" in str(err)

    def test_lexer_error_with_invalid_text(self):
        err = GCNLexerError("Unexpected character", invalid_text="@")
        msg = str(err)
        assert "Unexpected character" in msg
        assert "'@'" in msg

    def test_lexer_error_with_location(self):
        loc = ParseLocation(line=3, column=15, source_line="    alpha = @value;")
        err = GCNLexerError("Unexpected character", invalid_text="@", location=loc)
        msg = str(err)
        assert "line 3" in msg
        assert "@" in msg


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


class TestValidationError:
    def test_error_creation(self):
        err = ValidationError(message="Something wrong", severity="error")
        assert err.message == "Something wrong"
        assert err.severity == "error"
        assert err.is_error
        assert not err.is_warning

    def test_warning_creation(self):
        warn = ValidationError(message="Something suspicious", severity="warning")
        assert warn.is_warning
        assert not warn.is_error

    def test_to_exception(self):
        loc = ParseLocation(line=5, column=10)
        err = ValidationError(
            message="Undefined variable 'X'",
            severity="error",
            location=loc,
            suggestions=["Y"],
        )
        exc = err.to_exception()
        assert isinstance(exc, GCNSemanticError)
        assert "Undefined variable" in str(exc)

    def test_str_with_location(self):
        loc = ParseLocation(line=5, column=10)
        err = ValidationError(message="Bad thing", severity="error", location=loc)
        s = str(err)
        assert "[ERROR]" in s
        assert "line 5" in s
        assert "Bad thing" in s

    def test_str_without_location(self):
        err = ValidationError(message="Bad thing", severity="warning")
        s = str(err)
        assert "[WARNING]" in s
        assert "Bad thing" in s


class TestValidationErrorCollection:
    def test_empty_collection(self):
        coll = ValidationErrorCollection()
        assert len(coll) == 0
        assert not coll.has_errors
        assert coll.errors == []
        assert coll.warnings == []

    def test_add_error(self):
        coll = ValidationErrorCollection()
        coll.add_error("Something wrong")
        assert len(coll) == 1
        assert coll.has_errors
        assert len(coll.errors) == 1

    def test_add_warning(self):
        coll = ValidationErrorCollection()
        coll.add_warning("Something suspicious")
        assert len(coll) == 1
        assert not coll.has_errors
        assert len(coll.warnings) == 1

    def test_add_multiple(self):
        coll = ValidationErrorCollection()
        coll.add_error("Error 1")
        coll.add_warning("Warning 1")
        coll.add_error("Error 2")

        assert len(coll) == 3
        assert len(coll.errors) == 2
        assert len(coll.warnings) == 1

    def test_raise_first(self):
        coll = ValidationErrorCollection()
        coll.add_error("First error")
        coll.add_error("Second error")

        with pytest.raises(GCNSemanticError) as exc_info:
            coll.raise_first()
        assert "First error" in str(exc_info.value)

    def test_raise_first_skips_warnings(self):
        coll = ValidationErrorCollection()
        coll.add_warning("A warning")
        coll.add_error("The error")

        with pytest.raises(GCNSemanticError) as exc_info:
            coll.raise_first()
        assert "The error" in str(exc_info.value)

    def test_raise_first_no_errors(self):
        coll = ValidationErrorCollection()
        coll.add_warning("Just a warning")
        coll.raise_first()

    def test_raise_all_single(self):
        coll = ValidationErrorCollection()
        coll.add_error("Only error")

        with pytest.raises(GCNSemanticError) as exc_info:
            coll.raise_all()
        assert "Only error" in str(exc_info.value)

    def test_raise_all_multiple(self):
        coll = ValidationErrorCollection()
        coll.add_error("Error 1")
        coll.add_error("Error 2")
        coll.add_error("Error 3")

        with pytest.raises(GCNSemanticError) as exc_info:
            coll.raise_all()
        msg = str(exc_info.value)
        assert "Found 3 errors" in msg
        assert "Error 1" in msg
        assert "Error 2" in msg
        assert "Error 3" in msg

    def test_raise_all_no_errors(self):
        coll = ValidationErrorCollection()
        coll.add_warning("Just a warning")
        coll.raise_all()

    def test_iteration(self):
        coll = ValidationErrorCollection()
        coll.add_error("Error 1")
        coll.add_warning("Warning 1")

        issues = list(coll)
        assert len(issues) == 2

    def test_bool_empty(self):
        coll = ValidationErrorCollection()
        assert not coll

    def test_bool_non_empty(self):
        coll = ValidationErrorCollection()
        coll.add_warning("Something")
        assert coll

    def test_add_with_location_and_suggestions(self):
        coll = ValidationErrorCollection()
        loc = ParseLocation(line=10, column=5)
        coll.add_error("Undefined 'X'", location=loc, suggestions=["Y", "Z"])

        err = coll.errors[0]
        assert err.location == loc
        assert err.suggestions == ["Y", "Z"]
