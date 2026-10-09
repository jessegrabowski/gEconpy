import shutil
import subprocess

from collections.abc import Callable

import pytest
import sympy as sp

from gEconpy import model_from_gcn
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.exceptions import TypstPrintError
from gEconpy.model.latex import wrap_leads_in_expectations
from gEconpy.model.tables import CalibrationTable, EquationTable, SymbolRow, SymbolTable
from gEconpy.model.typst import typst
from tests._resources.cache_compiled_models import load_and_cache_example
from tests.conftest import TEST_GCNS

alpha = sp.Symbol("alpha")
A, C, K, mc = (TimeAwareSymbol(name, 0) for name in ("A", "C", "K", "mc"))


class TestNodes:
    @pytest.mark.parametrize(
        "expression, expected",
        [
            (C, "C_(t)"),
            (alpha, "alpha"),
            (mc, 'upright("mc")_(t)'),
            (sp.Integer(3), "3"),
            (sp.Float(0.99), "0.99"),
            (sp.Rational(1, 2), "frac(1, 2)"),
            (C + alpha, "alpha + C_(t)"),
            (C * alpha, "alpha C_(t)"),
            (C**alpha, "C_(t)^(alpha)"),
            (C / K, "frac(C_(t), K_(t))"),
            (C ** sp.Integer(-1), "frac(1, C_(t))"),
            (sp.log(A), "log(A_(t))"),
            (sp.exp(-C), "e^(-C_(t))"),
            ((C + alpha) * K, "K_(t) (alpha + C_(t))"),
            (
                TimeAwareSymbol("K", -1) ** alpha * TimeAwareSymbol("L", 0) ** (1 - alpha),
                "K_(t-1)^(alpha) L_(t)^(1 - alpha)",
            ),
        ],
        ids=[
            "time_aware_symbol",
            "parameter",
            "multi_letter",
            "integer",
            "float",
            "rational",
            "add",
            "mul",
            "pow",
            "fraction",
            "negative_power",
            "log",
            "exp",
            "parenthesized_sum",
            "cobb_douglas",
        ],
    )
    def test_each_node_renders(self, expression, expected):
        assert typst(expression) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [(1e-7, "1.0 times 10^(-7)"), (2.5e12, "2500000000000.0"), (0.99, "0.99")],
        ids=["scientific", "large_but_plain", "plain"],
    )
    def test_a_float_never_prints_pythons_exponent_notation(self, value, expected):
        """Typst reads the e as a variable, so 1.0e-7 typesets as 1.0e minus 7 rather than a small number."""
        assert typst(sp.Float(value)) == expected

    def test_an_expectation_uses_typsts_own_operator(self):
        """Typst has a double-struck E built in, so the operator needs no package the way LaTeX's does."""
        wrapped = wrap_leads_in_expectations(sp.Symbol("beta") * TimeAwareSymbol("U", 1))

        assert typst(wrapped) == "beta EE_t [U_(t+1)]"

    def test_a_node_with_no_rendering_raises(self):
        """A plausible-looking string for an unhandled node would put a silent error in a paper."""
        with pytest.raises(TypstPrintError, match=r"No Typst rendering for sin.*_print_sin method"):
            typst(sp.sin(C))

    def test_a_declared_name_replaces_the_inferred_one(self):
        assert typst(C + K, symbol_names={C: "cal(C)_(t)"}) == "cal(C)_(t) + K_(t)"


class TestAgreementWithLatex:
    """The two printers render the same expressions, so a structural divergence prints two different models."""

    def test_additive_terms_keep_the_order_the_latex_printer_gives_them(self):
        """Sympy's LaTeX printer reorders additive terms, and a printer that does not drifts from it."""
        assert sp.latex(mc - 1) == r"\text{mc}_{t} - 1"
        assert typst(mc - 1) == 'upright("mc")_(t) - 1'

    @pytest.mark.parametrize(
        "model_name", ["RBC", "RBC_two_household", "New_Keynesian"], ids=["rbc", "two_household", "nk"]
    )
    def test_every_sum_in_a_packaged_model_keeps_the_same_term_order(self, model_name):
        """
        Compare the order of each sum's terms, which is the divergence that matters.

        Both printers read one expression and sympy's LaTeX printer reorders what it is given. How either lays a
        term out is deliberately not compared: the LaTeX printer leaves a negative power as a leading factor
        where this one builds a ``frac``, which moves a variable within its own term and says the same thing.
        """
        model = load_and_cache_example(model_name)
        sums = {
            addition for equation in model.equations for addition in equation.atoms(sp.Add) if len(addition.args) > 1
        }
        assert sums, "a model with no sums cannot discriminate"

        for addition in sums:
            assert _term_order(addition, sp.latex) == _term_order(addition, typst), addition


def _term_order(addition: sp.Add, printer: Callable[[sp.Expr], str]) -> list[sp.Expr]:
    """Return the sum's terms, ordered by where the printer places each one in the rendered sum."""
    rendered = printer(addition)
    positions = {}
    for term in addition.args:
        # A negative term carries its sign into the sum as an operator, so the sign is not part of what to find.
        text = printer(term).lstrip("-").strip()
        assert text in rendered, f"{term} rendered as {text!r}, which is not in {rendered!r}"
        positions[term] = rendered.index(text)
    return sorted(positions, key=positions.__getitem__)


@pytest.fixture(scope="module")
def rbc():
    """Built once: every test here reads the same model, and building it dominates their runtime."""
    return load_and_cache_example("RBC")


class TestWriter:
    @pytest.mark.parametrize("group", ["variables", "shocks", "parameters"])
    def test_a_tabular_group_carries_no_latex(self, rbc, group):
        """A cell holding LaTeX would print its own backslashes in a Typst document."""
        markup = rbc.write_table(group, writer="typst")

        assert "\\text{" not in markup
        assert "\\linewidth" not in markup

    def test_the_symbols_are_rendered_in_typst(self, rbc):
        markup = rbc.write_table("variables", writer="typst")

        assert "[$lambda_(t)$]" in markup
        assert r"\lambda_{t}" not in markup

    def test_a_caption_and_a_label_wrap_the_table_in_a_figure(self, rbc):
        """Typst numbers and lists figures, and a label refers to one, so neither works on a bare table."""
        markup = rbc.write_table("variables", writer="typst", caption="Model variables", label="tab:vars")

        assert markup.startswith("#figure(")
        assert "table(" in markup
        assert "caption: [Model variables]," in markup
        assert markup.endswith("<tab:vars>")

    def test_a_table_with_neither_is_left_bare(self, rbc):
        assert rbc.write_table("variables", writer="typst").startswith("#table(")

    def test_a_size_sets_the_text_it_encloses(self, rbc):
        markup = rbc.write_table("variables", writer="typst", size="9pt")

        assert markup.startswith("#text(size: 9pt)[")
        assert markup.endswith("]")

    def test_a_size_encloses_the_figure_when_there_is_a_caption(self, rbc):
        """A size applies to the caption as much as to the table, so the figure goes inside it, not beside it."""
        markup = rbc.write_table("variables", writer="typst", size="9pt", caption="Variables")

        assert markup.startswith("#text(size: 9pt)[\n#figure(")
        assert markup.endswith("]")

    def test_widths_replace_the_defaults(self, rbc):
        markup = rbc.write_table("variables", writer="typst", widths=["20%", "60%"])

        assert "columns: (20%, 60%,)" in markup

    def test_a_wrong_number_of_widths_raises(self, rbc):
        with pytest.raises(ValueError, match="one width per printed column, got 3 for 2"):
            rbc.write_table("variables", writer="typst", widths=["20%", "60%", "10%"])

    def test_booktabs_replaces_the_grid_with_three_rules(self, rbc):
        """Typst draws a full grid by default, which is not what this literature prints."""
        ruled = rbc.write_table("variables", writer="typst", booktabs=True)

        assert "stroke: none" in ruled
        assert ruled.count("table.hline()") == 3
        assert "stroke: none" not in rbc.write_table("variables", writer="typst")

    def test_a_block_heading_precedes_its_own_math_block(self, rbc):
        """Typst has no intertext, so a heading is content between two blocks rather than a line inside one."""
        markup = rbc.write_table("equations", writer="typst")

        assert "*Household*\n\n$ " in markup
        assert "*Firm*\n\n$ " in markup

    def test_a_caption_is_separated_from_the_equation_it_labels(self, tmp_path):
        """A math block is sized to its content, so an h(1fr) leaves the caption jammed against the equation."""
        source = (TEST_GCNS / "one_block_2.gcn").read_text()
        labelled = source.replace(
            "        I[] = Y[] - C[] : lambda[];",
            '        @name = "Resource constraint" I[] = Y[] - C[] : lambda[];',
            1,
        )
        assert labelled != source, "one_block_2.gcn changed; update the replacement"
        path = tmp_path / "labelled.gcn"
        path.write_text(labelled)

        markup = model_from_gcn(path, verbose=False).write_table("equations", writer="typst")

        assert 'quad "Resource constraint"' in markup
        assert "#h(1fr)" not in markup

    def test_an_uncaptioned_first_order_condition_prints_its_derivative(self, rbc):
        markup = rbc.write_table("equations", writer="typst")

        assert "frac(partial cal(L), partial C_(t)) = 0 &=>" in markup

    def test_a_declared_typst_name_reaches_the_table(self, tmp_path):
        """A symbol declaring only ``latex`` has no Typst counterpart, so the printer infers one instead."""
        source = (TEST_GCNS / "open_rbc.gcn").read_text()
        declared = source.replace(
            "    C[] { positive = True; };",
            '    C[] { positive = True; latex = "\\mathcal{C}"; typst = "cal(C)"; };',
            1,
        )
        assert declared != source, "open_rbc.gcn changed; update the replacement"
        path = tmp_path / "declared.gcn"
        path.write_text(declared)
        model = model_from_gcn(path, verbose=False)

        markup = model.write_table("variables", writer="typst")

        assert "[$cal(C)_(t)$]" in markup
        assert "mathcal" not in markup

    def test_a_symbol_declaring_only_latex_falls_back_to_the_inferred_typst(self, tmp_path):
        """``C_(t)`` is also what an undeclared symbol renders as, so the LaTeX side proves an override exists."""
        source = (TEST_GCNS / "open_rbc.gcn").read_text()
        declared = source.replace(
            "    C[] { positive = True; };", '    C[] { positive = True; latex = "\\mathcal{C}"; };', 1
        )
        assert declared != source, "open_rbc.gcn changed; update the replacement"
        path = tmp_path / "latex_only.gcn"
        path.write_text(declared)
        model = model_from_gcn(path, verbose=False)

        assert r"${\mathcal{C}}_{t}$" in model.write_table("variables")
        assert "[$C_(t)$]" in model.write_table("variables", writer="typst")


class TestEmptyTable:
    """An empty table must not emit a figure wrapping nothing, which is the shape pdflatex once rejected."""

    @pytest.mark.parametrize(
        "table", [EquationTable([]), SymbolTable([]), CalibrationTable([])], ids=["equations", "symbols", "calibration"]
    )
    def test_an_empty_table_renders_as_nothing(self, table):
        assert table.to_typst() == ""

    @pytest.mark.parametrize(
        "style",
        [{}, {"caption": "Shocks"}, {"label": "tab:s"}, {"size": "9pt"}, {"caption": "Shocks", "label": "tab:s"}],
        ids=["bare", "caption", "label", "size", "caption_and_label"],
    )
    def test_an_empty_table_stays_empty_under_any_style(self, style):
        """A caption on no table still numbers one, so the wrapper has to drop with the body."""
        assert SymbolTable([]).to_typst(**style) == ""


class TestEscaping:
    @pytest.mark.parametrize(
        "character",
        ["#", "$", "@", "<", ">", "*", "_", "`", "[", "]"],
        ids=["hash", "dollar", "at", "less", "greater", "star", "underscore", "backtick", "open", "close"],
    )
    def test_a_markup_character_in_a_description_is_escaped(self, character):
        """Typst reads each of these as markup, so an unescaped one changes the document rather than printing."""
        table = SymbolTable([SymbolRow(symbol="C_(t)", description=f"a{character}b")])

        assert f"[a\\{character}b]" in table.to_typst()

    def test_a_backslash_is_escaped_before_the_escapes_it_would_double(self):
        """Escaping the backslash last would turn every escape this function had just added into a literal."""
        table = SymbolTable([SymbolRow(symbol="C_(t)", description="a\\b#c")])

        assert "[a\\\\b\\#c]" in table.to_typst()


def test_the_rows_can_be_built_for_typst_without_rendering_them(rbc):
    """``markup`` is the documented way to get Typst rows, and nothing else in the suite reaches it."""
    rows = rbc.table("variables", markup="typst").rows

    assert any(row.symbol == "lambda_(t)" for row in rows)
    assert not any("\\" in row.symbol for row in rows)


@pytest.mark.parametrize("group", ["equations", "variables", "parameters"])
def test_the_rendered_table_compiles(rbc, group, tmp_path):
    """
    The feature's whole contract is valid Typst, which only the compiler can confirm.

    ``typst`` is not a project dependency, so this skips where the binary is missing, exactly as the LaTeX
    compile test does for ``pdflatex``.
    """
    binary = shutil.which("typst")
    if binary is None:
        pytest.skip("typst not installed")

    source = tmp_path / "table.typ"
    source.write_text(rbc.write_table(group, writer="typst") + "\n")

    result = subprocess.run(
        [binary, "compile", source.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
