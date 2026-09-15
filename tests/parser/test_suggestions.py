import pytest

from gEconpy.parser.suggestions import (
    find_similar_names,
    suggest_assumption,
    suggest_block_component,
    suggest_distribution,
    suggest_wrapper,
)


class TestFindSimilarNames:
    @pytest.mark.parametrize(
        "name, candidates",
        [
            ("alpha", ["alpha", "beta"]),
            ("Alpha", ["alpha", "beta"]),
            ("alpha", {"Alpha", "Beta"}),
        ],
        ids=["exact", "query_uppercase", "candidate_uppercase"],
    )
    def test_exact_match_returns_empty(self, name, candidates):
        assert find_similar_names(name, candidates) == []

    @pytest.mark.parametrize(
        "name, candidates",
        [
            ("xyz", ["alpha", "beta"]),
            ("alpha", []),
        ],
        ids=["nothing_similar", "no_candidates"],
    )
    def test_no_match_returns_empty(self, name, candidates):
        assert find_similar_names(name, candidates) == []

    @pytest.mark.parametrize(
        "name, candidates, expected",
        [
            ("alpa", ["alpha", "beta", "gamma"], "alpha"),
            ("betta", ["alpha", "beta", "gamma"], "beta"),
            ("ahlpa", ["alpha", "beta", "gamma"], "alpha"),
            ("signa", {"alpha", "beta", "gamma", "delta", "epsilon", "theta", "sigma"}, "sigma"),
            ("Consumptin", {"Consumption", "Investment", "Output"}, "Consumption"),
            ("Caital", {"Capital", "Labor", "Output"}, "Capital"),
            ("Capitall", {"Capital", "Labor", "Output"}, "Capital"),
        ],
        ids=[
            "missing_letter",
            "doubled_letter",
            "transposition",
            "substitution",
            "missing_letter_long_name",
            "missing_letter_capitalized",
            "extra_letter",
        ],
    )
    def test_close_match_found(self, name, candidates, expected):
        assert expected in find_similar_names(name, candidates)

    def test_results_ranked_by_similarity(self):
        assert find_similar_names("bet", ["beta", "theta", "zeta"])[0] == "beta"

    def test_max_results_honored(self):
        assert len(find_similar_names("a", ["ab", "ac", "ad", "ae"], max_results=2)) == 2

    def test_min_similarity_filters(self):
        assert find_similar_names("abc", ["abcdef", "xyz"], min_similarity=0.5) == ["abcdef"]
        assert find_similar_names("abc", ["abcdef", "xyz"], min_similarity=0.8) == []


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Bet", "Beta"),
        ("Nomal", "Normal"),
        ("Gama", "Gamma"),
        ("TruncatdNormal", "TruncatedNormal"),
        ("Exponental", "Exponential"),
    ],
)
def test_suggest_distribution(name, expected):
    assert expected in suggest_distribution(name)


def test_suggest_distribution_ignores_case_of_exact_match():
    assert suggest_distribution("beta") == []


@pytest.mark.parametrize(
    "name, expected",
    [("Truncatd", "Truncated"), ("Censorred", "Censored"), ("maxen", "maxent")],
)
def test_suggest_wrapper(name, expected):
    assert expected in suggest_wrapper(name)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("defintions", "definitions"),
        ("calibraton", "calibration"),
        ("identites", "identities"),
        ("constrains", "constraints"),
        ("objectve", "objective"),
        ("contols", "controls"),
        ("schocks", "shocks"),
    ],
)
def test_suggest_block_component(name, expected):
    assert expected in suggest_block_component(name)


@pytest.mark.parametrize(
    "name, expected",
    [("positiv", "positive"), ("negtive", "negative"), ("nonnegtive", "nonnegative"), ("rea", "real")],
)
def test_suggest_assumption(name, expected):
    assert expected in suggest_assumption(name)
