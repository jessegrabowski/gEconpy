import pytest

from gEconpy.model.build import model_from_gcn
from gEconpy.parser.errors import GCNGrammarError, GCNParseError
from tests.conftest import ERROR_GCNS


@pytest.mark.parametrize(
    "gcn_file",
    ["E005_missing_rhs.gcn", "E007_unclosed_parenthesis.gcn", "E001_missing_semicolon.gcn"],
    ids=["missing_rhs", "unclosed_parenthesis", "missing_semicolon"],
)
def test_grammar_error_carries_location(gcn_file):
    with pytest.raises(GCNGrammarError) as exc:
        model_from_gcn(ERROR_GCNS / gcn_file, show_errors=False)
    assert exc.value.location is not None


def test_show_errors_prints_to_stderr(capsys):
    with pytest.raises(GCNParseError):
        model_from_gcn(ERROR_GCNS / "E005_missing_rhs.gcn", show_errors=True)
    assert "error" in capsys.readouterr().err.lower()


def test_show_errors_false_is_silent(capsys):
    with pytest.raises(GCNParseError):
        model_from_gcn(ERROR_GCNS / "E005_missing_rhs.gcn", show_errors=False)
    assert capsys.readouterr().err == ""
