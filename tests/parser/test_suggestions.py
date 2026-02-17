import pytest

from gEconpy.parser.suggestions import (
    find_similar_names,
    suggest_assumption,
    suggest_block_component,
    suggest_distribution,
    suggest_wrapper,
)


class TestFindSimilarNames:
    def test_exact_match_returns_empty(self):
        result = find_similar_names("alpha", ["alpha", "beta"])
        assert result == []

    def test_exact_match_case_insensitive(self):
        result = find_similar_names("Alpha", ["alpha", "beta"])
        assert result == []

    def test_close_match_found(self):
        result = find_similar_names("alpa", ["alpha", "beta", "gamma"])
        assert "alpha" in result

    def test_multiple_close_matches_ranked(self):
        result = find_similar_names("bet", ["beta", "theta", "zeta"])
        assert result[0] == "beta"

    def test_no_match_returns_empty(self):
        result = find_similar_names("xyz", ["alpha", "beta"])
        assert result == []

    def test_max_results_honored(self):
        result = find_similar_names("a", ["ab", "ac", "ad", "ae"], max_results=2)
        assert len(result) <= 2

    def test_min_similarity_filters(self):
        result = find_similar_names("abc", ["abcdef", "xyz"], min_similarity=0.8)
        # "abc" vs "abcdef" has ~0.67 similarity, should be filtered at 0.8
        assert "xyz" not in result

    def test_empty_candidates_returns_empty(self):
        result = find_similar_names("alpha", [])
        assert result == []

    def test_single_character_difference(self):
        result = find_similar_names("betta", ["alpha", "beta", "gamma"])
        assert "beta" in result

    def test_transposition(self):
        result = find_similar_names("ahlpa", ["alpha", "beta", "gamma"])
        assert "alpha" in result

    def test_suggests_similar_parameter(self):
        known = {"alpha", "beta", "gamma", "delta"}
        suggestions = find_similar_names("alpa", known)
        assert "alpha" in suggestions

    def test_parameter_case_insensitive(self):
        known = {"Alpha", "Beta"}
        suggestions = find_similar_names("alpha", known)
        assert suggestions == []

    def test_parameter_no_match(self):
        known = {"alpha", "beta"}
        suggestions = find_similar_names("xyz", known)
        assert suggestions == []

    def test_typo_in_greek_letter(self):
        known = {"alpha", "beta", "gamma", "delta", "epsilon", "theta", "sigma"}
        suggestions = find_similar_names("signa", known)
        assert "sigma" in suggestions

    def test_suggests_similar_variable(self):
        known = {"Consumption", "Investment", "Output"}
        suggestions = find_similar_names("Consumptin", known)
        assert "Consumption" in suggestions

    def test_variable_missing_letter(self):
        known = {"Capital", "Labor", "Output"}
        suggestions = find_similar_names("Caital", known)
        assert "Capital" in suggestions

    def test_variable_extra_letter(self):
        known = {"Capital", "Labor", "Output"}
        suggestions = find_similar_names("Capitall", known)
        assert "Capital" in suggestions


class TestSuggestDistribution:
    def test_suggests_known_distribution(self):
        suggestions = suggest_distribution("Bet")
        assert "Beta" in suggestions

    def test_suggests_normal(self):
        suggestions = suggest_distribution("Nomal")
        assert "Normal" in suggestions

    def test_suggests_gamma(self):
        suggestions = suggest_distribution("Gama")
        assert "Gamma" in suggestions

    def test_case_variation(self):
        suggestions = suggest_distribution("beta")
        # Exact match case-insensitive
        assert suggestions == []

    def test_truncated_normal(self):
        suggestions = suggest_distribution("TruncatdNormal")
        assert "TruncatedNormal" in suggestions

    def test_exponential(self):
        suggestions = suggest_distribution("Exponental")
        assert "Exponential" in suggestions


class TestSuggestWrapper:
    def test_suggests_truncated(self):
        suggestions = suggest_wrapper("Truncatd")
        assert "Truncated" in suggestions

    def test_suggests_censored(self):
        suggestions = suggest_wrapper("Censorred")
        assert "Censored" in suggestions

    def test_suggests_maxent(self):
        suggestions = suggest_wrapper("maxen")
        assert "maxent" in suggestions


class TestSuggestBlockComponent:
    def test_suggests_definitions(self):
        suggestions = suggest_block_component("defintions")
        assert "definitions" in suggestions

    def test_suggests_calibration(self):
        suggestions = suggest_block_component("calibraton")
        assert "calibration" in suggestions

    def test_suggests_identities(self):
        suggestions = suggest_block_component("identites")
        assert "identities" in suggestions

    def test_suggests_constraints(self):
        suggestions = suggest_block_component("constrains")
        assert "constraints" in suggestions

    def test_suggests_objective(self):
        suggestions = suggest_block_component("objectve")
        assert "objective" in suggestions

    def test_suggests_controls(self):
        suggestions = suggest_block_component("contols")
        assert "controls" in suggestions

    def test_suggests_shocks(self):
        suggestions = suggest_block_component("schocks")
        assert "shocks" in suggestions


class TestSuggestAssumption:
    def test_suggests_positive(self):
        suggestions = suggest_assumption("positiv")
        assert "positive" in suggestions

    def test_suggests_negative(self):
        suggestions = suggest_assumption("negtive")
        assert "negative" in suggestions

    def test_suggests_nonnegative(self):
        suggestions = suggest_assumption("nonnegtive")
        assert "nonnegative" in suggestions

    def test_suggests_real(self):
        suggestions = suggest_assumption("rea")
        assert "real" in suggestions
