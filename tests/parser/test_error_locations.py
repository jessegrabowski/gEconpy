import pytest

from gEconpy.parser.errors import GCNGrammarError
from gEconpy.parser.preprocessor import preprocess


@pytest.mark.parametrize(
    "source, line, column",
    [
        ("block TEST { identities { Y[] = ; }; };", 1, 27),
        ("block TEST { identities { = C[]; }; };", 1, 27),
        ("block TEST { identities { Y[] = C[]; }", 1, 39),
        ("block TEST\n{\n    identities\n    {\n        Y[] = log(C[];\n    };\n};", 5, 22),
        ("block TEST\n{\n    identities { Y[] = C[]) + I[]; };\n};", 3, 27),
        ("block TEST\n{\n    identities { Y[] = log(C[] + I[]; };\n};", 3, 37),
        ("block TEST\n{\n    identities { Y[abc] = C[]; };\n};", 3, 18),
        ("block TEST\n{\n    identities { Y[] = C[]; };\n    calibration\n    {\n        alpha = ;\n    };\n};", 6, 9),
        (
            "block FIRST\n{\n    identities { Y[] = C[]; };\n};\n\nblock SECOND\n{\n    identities { X[] = (); };\n};",
            8,
            25,
        ),
    ],
    ids=[
        "empty_rhs",
        "empty_lhs",
        "unclosed_block_brace",
        "unclosed_parenthesis",
        "extra_closing_parenthesis",
        "unclosed_function_call",
        "invalid_time_index",
        "empty_calibration_rhs",
        "error_in_second_block",
    ],
)
def test_grammar_error_reports_location(source, line, column):
    with pytest.raises(GCNGrammarError) as exc_info:
        preprocess(source, validate=True)

    location = exc_info.value.location
    assert location is not None
    assert (location.line, location.column) == (line, column)


@pytest.mark.parametrize(
    "source",
    [
        "block TEST\n{\n    identities { Y[] = C[] + + I[]; };\n};",
        "block TEST\n{\n    identities { Y[] = log(); };\n};",
    ],
    ids=["double_plus", "empty_function_arguments"],
)
def test_malformed_expression_rejected(source):
    with pytest.raises(GCNGrammarError):
        preprocess(source, validate=True)
