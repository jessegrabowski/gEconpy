import pytest
import sympy as sp

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol, render_latex


class TestTimeSubscript:
    @pytest.mark.parametrize(
        "time_index, expected",
        [(0, "K_{t}"), (-1, "K_{t-1}"), (1, "K_{t+1}"), (-2, "K_{t-2}"), (2, "K_{t+2}"), ("ss", "K_{ss}")],
        ids=["t", "lag", "lead", "two_lags", "two_leads", "steady_state"],
    )
    def test_the_time_index_becomes_the_subscript(self, time_index, expected):
        assert sp.latex(TimeAwareSymbol("K", time_index)) == expected

    def test_a_declared_subscript_keeps_the_time_index_beside_it(self):
        """Sympy's own rendering joins the two with a space, which reads as one two-letter subscript."""
        assert sp.latex(TimeAwareSymbol("sigma_C", 0)) == r"\sigma_{C,t}"


class TestStemInference:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("alpha", r"\alpha_{t}"),
            ("beta", r"\beta_{t}"),
            ("rho", r"\rho_{t}"),
            ("Theta", r"\Theta_{t}"),
            ("Lambda", r"\Lambda_{t}"),
            ("Omega", r"\Omega_{t}"),
        ],
        ids=["alpha", "beta", "rho", "upper_theta", "upper_lambda", "upper_omega"],
    )
    def test_a_greek_name_becomes_its_letter(self, name, expected):
        assert sp.latex(TimeAwareSymbol(name, 0)) == expected

    @pytest.mark.parametrize("name", ["epsilon", "epsilon_A"], ids=["bare", "with_subscript"])
    def test_epsilon_renders_as_varepsilon(self, name):
        r"""The macro literature writes shocks with \\varepsilon, and sympy's default is the other glyph."""
        assert sp.latex(TimeAwareSymbol(name, 0)).startswith(r"\varepsilon")

    @pytest.mark.parametrize("name", ["Epsilon", "Eta", "Iota"], ids=["epsilon", "eta", "iota"])
    def test_a_greek_name_with_no_uppercase_command_falls_back_to_text(self, name):
        """Their capitals are ordinary Latin letters, so there is no command to emit."""
        assert sp.latex(TimeAwareSymbol(name, 0)) == rf"\text{{{name}}}_{{t}}"

    @pytest.mark.parametrize(
        "name, expected",
        [("mc", r"\text{mc}_{t}"), ("Div", r"\text{Div}_{t}"), ("TFP", r"\text{TFP}_{t}")],
        ids=["marginal_cost", "dividends", "acronym"],
    )
    def test_a_multi_letter_name_is_wrapped_so_it_does_not_read_as_a_product(self, name, expected):
        assert sp.latex(TimeAwareSymbol(name, 0)) == expected

    @pytest.mark.parametrize("name", ["K", "C", "Y"], ids=["capital", "consumption", "output"])
    def test_a_single_letter_name_is_left_alone(self, name):
        assert sp.latex(TimeAwareSymbol(name, 0)) == rf"{name}_{{t}}"


def test_sympy_renders_the_expression_structure_around_the_symbols():
    """The point of rendering at the symbol is that sympy handles everything else without help."""
    beta = sp.Symbol("beta")
    lam, r = TimeAwareSymbol("lambda", 1), TimeAwareSymbol("r", 1)
    delta = sp.Symbol("delta")

    rendered = sp.latex(beta * (lam * r - lam * (delta - 1)) - TimeAwareSymbol("lambda", 0))

    assert rendered == (
        r"\beta \left(\lambda_{t+1} r_{t+1} - \lambda_{t+1} \left(\delta - 1\right)\right) - \lambda_{t}"
    )


class TestOverride:
    def test_an_override_replaces_the_stem_and_keeps_the_subscripts(self):
        symbol = TimeAwareSymbol("mc", -1)

        assert render_latex(symbol, stem_override=r"\mathcal{M}") == r"\mathcal{M}_{t-1}"

    def test_an_override_composes_with_a_declared_subscript(self):
        symbol = TimeAwareSymbol("sigma_C", 1)

        assert render_latex(symbol, stem_override=r"\varsigma") == r"\varsigma_{C,t+1}"

    def test_sympys_symbol_names_setting_reaches_a_time_aware_symbol(self):
        """Sympy dispatches to ``_latex`` before ``_print_Symbol``, so the setting only works if _latex reads it."""
        symbol = TimeAwareSymbol("mc", 0)

        assert sp.latex(symbol, symbol_names={symbol: r"\mathcal{M}_{t}"}) == r"\mathcal{M}_{t}"

    def test_a_symbol_with_no_override_still_infers_its_own(self):
        overridden, plain = TimeAwareSymbol("mc", 0), TimeAwareSymbol("Div", 0)

        rendered = sp.latex(overridden + plain, symbol_names={overridden: r"\mathcal{M}_{t}"})

        assert r"\mathcal{M}_{t}" in rendered
        assert r"\text{Div}_{t}" in rendered
