import pytest

from gEconpy.classes.time_aware_symbol import (
    TimeAwareSymbol,
    render_latex,
    render_name_typst,
    render_typst,
)


class TestStem:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("alpha", "alpha"),
            ("Gamma", "Gamma"),
            ("epsilon", "epsilon"),
            ("C", "C"),
            ("mc", 'upright("mc")'),
            ("TFP", 'upright("TFP")'),
            ("omicron", 'upright("omicron")'),
        ],
        ids=["greek", "capital_greek", "epsilon", "single_letter", "multi_letter", "acronym", "no_such_letter"],
    )
    def test_a_stem_renders_by_what_typst_already_knows(self, name, expected):
        """Typst has the greek letters built in, so the command table LaTeX needs is only a membership test here."""
        assert render_name_typst(name) == expected

    @pytest.mark.parametrize(
        "time_index, expected",
        [(0, "C_(t)"), (-1, "C_(t-1)"), (1, "C_(t+1)"), ("ss", "C_(ss)")],
        ids=["contemporaneous", "lag", "lead", "steady_state"],
    )
    def test_the_time_index_becomes_the_last_subscript(self, time_index, expected):
        assert render_typst(TimeAwareSymbol("C", time_index)) == expected

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("epsilon_beta", "epsilon_(beta,t)"),
            ("shock_preference", 'upright("shock")_(upright("preference"),t)'),
            ("mc_hat_x", 'upright("mc")_(upright("hat"),x,t)'),
        ],
        ids=["greek_subscript", "word_subscript", "two_underscores"],
    )
    def test_each_subscript_is_rendered_by_the_same_rules_as_the_stem(self, name, expected):
        assert render_typst(TimeAwareSymbol(name, 0)) == expected


class TestOverride:
    def test_an_override_without_a_subscript_takes_the_time_index(self):
        assert render_typst(TimeAwareSymbol("mc", -1), name_override="cal(M)") == "cal(M)_(t-1)"

    def test_an_override_replaces_the_declared_subscripts_too(self):
        assert render_typst(TimeAwareSymbol("sigma_C", 1), name_override="varsigma") == "varsigma_(t+1)"

    @pytest.mark.parametrize("override", ["hat(p)_H", "hat(p)_(H)"], ids=["bare_subscript", "bracketed_subscript"])
    def test_the_time_index_joins_a_subscript_the_override_ends_with(self, override):
        assert render_typst(TimeAwareSymbol("p_H_tilde", 0), name_override=override) == "hat(p)_(H,t)"

    def test_an_override_is_not_wrapped_the_way_latex_wraps_one(self):
        """LaTeX braces the override to avoid a double subscript. Typst binds the subscript to one atom instead."""
        symbol = TimeAwareSymbol("mc", 0)

        assert render_latex(symbol, name_override=r"\mathcal{M}") == r"{\mathcal{M}}_{t}"
        assert render_typst(symbol, name_override="cal(M)") == "cal(M)_(t)"
