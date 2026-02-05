import pytest

from gEconpy.parser.constants import DEFAULT_ASSUMPTIONS
from gEconpy.parser.grammar.special_blocks import (
    extract_special_block_content,
    parse_assumptions,
    parse_options,
    parse_tryreduce,
    remove_special_block,
)


class TestParseOptions:
    def test_empty_options(self):
        text = "options { };"
        result = parse_options(text)
        assert result == {}

    def test_single_option_true(self):
        text = "options { output logfile = TRUE; };"
        result = parse_options(text)
        assert result == {"output logfile": True}

    def test_single_option_false(self):
        text = "options { output logfile = FALSE; };"
        result = parse_options(text)
        assert result == {"output logfile": False}

    def test_multiple_options(self):
        text = """options
        {
            output logfile = TRUE;
            output LaTeX = TRUE;
            output LaTeX landscape = TRUE;
        };"""
        result = parse_options(text)
        assert result == {
            "output logfile": True,
            "output LaTeX": True,
            "output LaTeX landscape": True,
        }

    def test_mixed_true_false(self):
        text = "options { output logfile = FALSE; output LaTeX = TRUE; };"
        result = parse_options(text)
        assert result["output logfile"] is False
        assert result["output LaTeX"] is True

    def test_no_options_block(self):
        text = "block HOUSEHOLD { };"
        result = parse_options(text)
        assert result == {}

    def test_case_insensitive_true_false(self):
        text = "options { a = true; b = True; c = TRUE; };"
        result = parse_options(text)
        assert all(v is True for v in result.values())


class TestParseTryreduce:
    def test_empty_tryreduce(self):
        text = "tryreduce { };"
        result = parse_tryreduce(text)
        assert result == []

    def test_single_variable(self):
        text = "tryreduce { U[]; };"
        result = parse_tryreduce(text)
        assert result == ["U[]"]

    def test_multiple_variables(self):
        text = "tryreduce { U[], TC[]; };"
        result = parse_tryreduce(text)
        assert set(result) == {"U[]", "TC[]"}

    def test_variables_with_newlines(self):
        text = """tryreduce
        {
            U[], TC[];
        };"""
        result = parse_tryreduce(text)
        assert set(result) == {"U[]", "TC[]"}

    def test_no_tryreduce_block(self):
        text = "block HOUSEHOLD { };"
        result = parse_tryreduce(text)
        assert result == []

    def test_from_one_block_ss(self):
        # Pattern from one_block_1_ss.gcn
        text = "tryreduce { C[]; };"
        result = parse_tryreduce(text)
        assert result == ["C[]"]


class TestParseAssumptions:
    def test_no_assumptions_block_returns_defaults(self):
        text = "block HOUSEHOLD { };"
        result = parse_assumptions(text)
        assert result["any_var"] == DEFAULT_ASSUMPTIONS

    def test_empty_assumptions_block(self):
        text = "assumptions { };"
        result = parse_assumptions(text)
        assert result["any_var"] == DEFAULT_ASSUMPTIONS

    def test_positive_assumptions(self):
        text = """assumptions
        {
            positive
            {
                C[], K[], L[];
            };
        };"""
        result = parse_assumptions(text)
        assert result["C"] == {"real": True, "positive": True}
        assert result["K"] == {"real": True, "positive": True}
        assert result["L"] == {"real": True, "positive": True}

    def test_negative_assumptions(self):
        text = """assumptions
        {
            negative
            {
                TC[];
            };
        };"""
        result = parse_assumptions(text)
        assert result["TC"]["negative"] is True

    def test_mixed_assumptions(self):
        text = """assumptions
        {
            negative
            {
                TC[];
            };
            positive
            {
                C[], K[];
            };
        };"""
        result = parse_assumptions(text)
        assert result["TC"]["negative"] is True
        assert result["C"]["positive"] is True
        assert result["K"]["positive"] is True

    def test_invalid_assumption_raises(self):
        text = """assumptions
        {
            random_words
            {
                L[], M[], P[];
            };
        };"""
        with pytest.raises(ValueError, match="not a valid Sympy assumption"):
            parse_assumptions(text)

    def test_typo_gives_suggestion(self):
        text = """assumptions
        {
            possitive
            {
                L[], M[], P[];
            };
        };"""
        with pytest.raises(ValueError, match='Did you mean "positive"'):
            parse_assumptions(text)

    def test_parameters_without_brackets(self):
        text = """assumptions
        {
            positive
            {
                C[], K[], L[], A[], lambda[], w[], r[], mc[],
                beta, delta, sigma_C, sigma_L, alpha;
            };
        };"""
        result = parse_assumptions(text)
        assert result["beta"]["positive"] is True
        assert result["C"]["positive"] is True

    def test_full_nk_assumptions(self):
        # Pattern from full_nk.gcn
        text = """assumptions
        {
            negative
            {
                TC[];
            };
            positive
            {
                shock_technology[], shock_preference[], pi[], pi_star[], pi_obj[], r[], r_G[], mc[], w[], w_star[],
                Y[], C[], I[], K[], L[],
                delta, beta, sigma_C, sigma_L, gamma_I, phi_H;
            };
        };"""
        result = parse_assumptions(text)
        assert result["TC"]["negative"] is True
        assert result["Y"]["positive"] is True
        assert result["beta"]["positive"] is True


class TestExtractSpecialBlockContent:
    def test_extract_options(self):
        text = "options { output logfile = TRUE; }; block HOUSEHOLD { };"
        content = extract_special_block_content(text, "options")
        assert "output logfile" in content

    def test_extract_tryreduce(self):
        text = "tryreduce { U[], TC[]; }; block HOUSEHOLD { };"
        content = extract_special_block_content(text, "tryreduce")
        assert "U[]" in content

    def test_extract_assumptions(self):
        text = "assumptions { positive { C[]; }; }; block HOUSEHOLD { };"
        content = extract_special_block_content(text, "assumptions")
        assert "positive" in content

    def test_nonexistent_block(self):
        text = "block HOUSEHOLD { };"
        content = extract_special_block_content(text, "options")
        assert content is None


class TestRemoveSpecialBlock:
    def test_remove_options(self):
        text = "options { output logfile = TRUE; }; block HOUSEHOLD { };"
        result = remove_special_block(text, "options")
        assert "options" not in result
        assert "HOUSEHOLD" in result

    def test_remove_tryreduce(self):
        text = "tryreduce { U[]; }; block HOUSEHOLD { };"
        result = remove_special_block(text, "tryreduce")
        assert "tryreduce" not in result
        assert "HOUSEHOLD" in result

    def test_remove_nonexistent_block(self):
        text = "block HOUSEHOLD { };"
        result = remove_special_block(text, "options")
        assert result == text


class TestRealWorldPatterns:
    """Test patterns from actual GCN files."""

    def test_full_nk_options(self):
        text = """options
        {
            output logfile = TRUE;
            output LaTeX = TRUE;
            output LaTeX landscape = TRUE;
        };"""
        result = parse_options(text)
        assert result == {
            "output logfile": True,
            "output LaTeX": True,
            "output LaTeX landscape": True,
        }

    def test_rbc_tryreduce(self):
        text = """tryreduce
        {
            U[], TC[];
        };"""
        result = parse_tryreduce(text)
        assert set(result) == {"U[]", "TC[]"}

    def test_combined_special_blocks(self):
        text = """options
        {
            output logfile = TRUE;
        };

        tryreduce
        {
            U[];
        };

        assumptions
        {
            positive
            {
                C[], K[];
            };
        };

        block HOUSEHOLD
        {
            identities { Y[] = C[]; };
        };"""

        options = parse_options(text)
        tryreduce = parse_tryreduce(text)
        assumptions = parse_assumptions(text)

        assert options == {"output logfile": True}
        assert tryreduce == ["U[]"]
        assert assumptions["C"]["positive"] is True
        assert assumptions["K"]["positive"] is True

        # Remove all special blocks
        remaining = text
        for block_name in ["options", "tryreduce", "assumptions"]:
            remaining = remove_special_block(remaining, block_name)

        assert "options" not in remaining
        assert "tryreduce" not in remaining
        assert "assumptions" not in remaining
        assert "HOUSEHOLD" in remaining
