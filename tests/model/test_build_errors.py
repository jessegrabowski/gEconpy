from pathlib import Path

import pytest

from gEconpy.model.build import model_from_gcn
from gEconpy.parser.errors import GCNGrammarError, GCNParseError


@pytest.fixture
def error_gcns_dir():
    return Path(__file__).parent.parent / "_resources" / "error_gcns"


class TestModelBuildErrors:
    def test_missing_rhs_raises_grammar_error(self, error_gcns_dir):
        gcn_path = error_gcns_dir / "E005_missing_rhs.gcn"
        with pytest.raises(GCNGrammarError) as exc:
            model_from_gcn(gcn_path, show_errors=False)
        assert exc.value.location is not None

    def test_unbalanced_parens_raises_grammar_error(self, error_gcns_dir):
        gcn_path = error_gcns_dir / "E007_unclosed_parenthesis.gcn"
        with pytest.raises(GCNGrammarError) as exc:
            model_from_gcn(gcn_path, show_errors=False)
        assert exc.value.location is not None

    def test_missing_semicolon_raises_grammar_error(self, error_gcns_dir):
        gcn_path = error_gcns_dir / "E001_missing_semicolon.gcn"
        with pytest.raises(GCNGrammarError) as exc:
            model_from_gcn(gcn_path, show_errors=False)
        assert exc.value.location is not None


class TestShowErrorsFlag:
    def test_show_errors_true_prints_to_stderr(self, error_gcns_dir, capsys):
        gcn_path = error_gcns_dir / "E005_missing_rhs.gcn"
        with pytest.raises(GCNParseError):
            model_from_gcn(gcn_path, show_errors=True)
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()

    def test_show_errors_false_silent(self, error_gcns_dir, capsys):
        gcn_path = error_gcns_dir / "E005_missing_rhs.gcn"
        with pytest.raises(GCNParseError):
            model_from_gcn(gcn_path, show_errors=False)
        captured = capsys.readouterr()
        assert captured.err == ""
