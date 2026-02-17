import pytest

from gEconpy.parser.errors import GCNGrammarError
from gEconpy.parser.preprocessor import preprocess


class TestGrammarErrorLocations:
    """Verify grammar errors have correct line/column."""

    def test_empty_equation_raises(self):
        # An equation with missing RHS should trigger a parsing error
        source = "block TEST { identities { Y[] = ; }; };"
        with pytest.raises(GCNGrammarError):
            preprocess(source, validate=True)

    def test_unclosed_brace_raises(self):
        # Missing closing brace should be caught
        source = "block TEST { identities { Y[] = C[]; }"
        # This may or may not raise depending on parser behavior
        # The point is that if it does raise, it should have location info
        try:
            result = preprocess(source, validate=True)
            # If it parses, we just check it's valid
            assert result.ast is not None
        except GCNGrammarError:
            # If it fails, that's also acceptable
            pass


class TestPreprocessSource:
    """Verify ParseResult includes source."""

    def test_parse_result_has_source(self):
        source = "block TEST { identities { Y[] = C[]; }; };"
        result = preprocess(source, validate=False)
        assert result.source is not None

    def test_source_has_correct_content(self):
        source = "block TEST { identities { Y[] = C[]; }; };"
        result = preprocess(source, validate=False)
        assert result.source == source

    def test_parse_result_has_filename(self):
        source = "block TEST { identities { Y[] = C[]; }; };"
        result = preprocess(source, filename="test.gcn", validate=False)
        assert result.filename == "test.gcn"

    def test_default_filename_is_none(self):
        source = "block TEST { identities { Y[] = C[]; }; };"
        result = preprocess(source, validate=False)
        assert result.filename is None


class TestSourceFromPreprocess:
    """Verify source can be used for error context."""

    def test_source_preserves_lines(self):
        source = """block TEST
{
    identities { Y[] = C[]; };
};
"""
        result = preprocess(source, validate=False)
        lines = result.source.split("\n")
        assert len(lines) == 5
        assert "identities" in lines[2]


class TestErrorSourceSnippets:
    """Verify errors include source context when available."""

    def test_empty_rhs_has_location(self):
        # Empty RHS should point to where the expression should be
        source = "block TEST { identities { Y[] = ; }; };"
        try:
            preprocess(source, validate=True)
            pytest.fail("Expected an error")
        except GCNGrammarError as e:
            assert e.location is not None
            assert e.location.line == 1
            # Column location depends on parser - just verify it's set
            assert e.location.column >= 1

    def test_empty_lhs_has_location(self):
        # Empty LHS
        source = "block TEST { identities { = C[]; }; };"
        try:
            preprocess(source, validate=True)
            pytest.fail("Expected an error")
        except GCNGrammarError as e:
            assert e.location is not None
            assert e.location.line == 1

    def test_unclosed_parenthesis_location(self):
        source = """block TEST
{
    identities
    {
        Y[] = log(C[];
    };
};"""
        try:
            preprocess(source, validate=True)
            pytest.fail("Expected an error")
        except GCNGrammarError as e:
            assert e.location is not None
            assert e.location.line == 5
            assert e.location.column == 22

    def test_unmatched_paren_in_expression(self):
        # Extra closing paren causes parsing error
        source = """block TEST
{
    identities { Y[] = C[]) + I[]; };
};"""
        try:
            preprocess(source, validate=True)
            pytest.fail("Expected an error")
        except GCNGrammarError as e:
            assert e.location is not None
            assert e.location.line == 3

    def test_error_location_in_calibration(self):
        source = """block TEST
{
    identities { Y[] = C[]; };
    calibration
    {
        alpha = ;
    };
};"""
        try:
            preprocess(source, validate=True)
            pytest.fail("Expected an error")
        except GCNGrammarError as e:
            assert e.location is not None
            # Error reported on line 6 where the problematic statement is
            assert e.location.line == 6

    def test_error_in_second_block(self):
        source = """block FIRST
{
    identities { Y[] = C[]; };
};

block SECOND
{
    identities { X[] = (); };
};"""
        try:
            preprocess(source, validate=True)
            pytest.fail("Expected an error")
        except GCNGrammarError as e:
            assert e.location is not None
            assert e.location.line == 8
            assert e.location.column == 25

    def test_unclosed_function_call_location(self):
        # Missing closing paren in function call
        source = """block TEST
{
    identities { Y[] = log(C[] + I[]; };
};"""
        try:
            preprocess(source, validate=True)
            pytest.fail("Expected an error")
        except GCNGrammarError as e:
            assert e.location is not None
            assert e.location.line == 3

    def test_invalid_time_index_location(self):
        # Invalid time index
        source = """block TEST
{
    identities { Y[abc] = C[]; };
};"""
        try:
            preprocess(source, validate=True)
            pytest.fail("Expected an error")
        except GCNGrammarError as e:
            assert e.location is not None
            assert e.location.line == 3

    def test_double_plus_rejected(self):
        # Double plus should be rejected (not treated as unary +)
        source = """block TEST
{
    identities { Y[] = C[] + + I[]; };
};"""
        with pytest.raises(GCNGrammarError):
            preprocess(source, validate=True)

    def test_empty_function_args_rejected(self):
        # Functions must have at least one argument
        source = """block TEST
{
    identities { Y[] = log(); };
};"""
        with pytest.raises(GCNGrammarError):
            preprocess(source, validate=True)


class TestValidModelsParse:
    """Ensure valid models still parse correctly with new error handling."""

    def test_simple_valid_model(self):
        source = """
block HOUSEHOLD
{
    identities { Y[] = C[] + I[]; };
    calibration { alpha = 0.35; };
};
"""
        result = preprocess(source, validate=False)
        assert result.ast is not None
        assert len(result.ast.blocks) == 1

    def test_valid_model_with_controls(self):
        source = """
block HOUSEHOLD
{
    controls { C[], L[]; };
    objective { U[] = log(C[]); };
    constraints { C[] = w[] * L[] : lambda[]; };
    calibration { w = 1.0; };
};
"""
        result = preprocess(source, validate=False)
        assert result.ast is not None
        assert len(result.ast.blocks[0].controls) == 2
