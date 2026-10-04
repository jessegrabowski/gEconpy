import re

import pytest

from gEconpy import model_from_gcn
from gEconpy.data import get_example_gcn
from gEconpy.model.html import SOLVED_STEADY_STATE_TITLE, render_gcn_file, render_model
from gEconpy.model.latex import block_heading
from tests.conftest import TEST_GCNS

# Sphinx tells MathJax to re-enter an ignored subtree only for these, so the container must carry one of them.
SPHINX_TYPESET_CLASSES = ("tex2jax_process", "mathjax_process", "math", "output_area")

# Variables, shocks and parameters each get a section of their own before the authored blocks.
SUMMARY_SECTIONS = 3


@pytest.fixture(scope="module")
def rendered():
    """Render once: parsing the file dominates the runtime of every assertion here."""
    return render_gcn_file(get_example_gcn("RBC"))


@pytest.fixture(scope="module")
def rbc():
    """Build once: building the model dominates the runtime of every assertion below."""
    return model_from_gcn(get_example_gcn("RBC"), verbose=False)


def test_the_render_runs_no_javascript(rendered):
    """A frontend that defines window.MathJax without a Hub left the old polling loop running forever."""
    assert "<script" not in rendered
    assert "cdnjs.cloudflare.com" not in rendered


def test_the_container_carries_a_class_sphinx_typesets(rendered):
    """A myst-nb page sets tex2jax_ignore on its section, and MathJax re-enters only these classes."""
    container = next(line for line in rendered.splitlines() if "ge-model" in line and "<div" in line)
    classes = re.search(r"class='([^']*)'", container).group(1).split()

    assert set(classes) & set(SPHINX_TYPESET_CLASSES), container


def test_the_equations_carry_display_delimiters(rendered):
    r"""``\[...\]`` is the one display form every frontend recognizes, and ``$...$`` is not."""
    assert r"\[" in rendered
    assert r"\]" in rendered


def test_two_renders_in_one_document_cannot_collide(rbc, rendered):
    """An id selector made two models in one notebook share whichever stylesheet matched first."""
    document = rendered + rbc._repr_html_()

    assert document.count("class='ge-model") == 2
    assert not re.findall(r"\n\s*#[\w-]+[\s,{]", document)


def test_an_unbuilt_file_shows_the_same_view_as_a_built_model(rendered):
    """``print_gcn_file`` used to render a component-headings layout with no program and no styling."""
    assert r"\max_{K_{t-1}, L_{t}}" in rendered
    assert "subject to" in rendered
    assert r"\quad (\text{mc}_{t})" in rendered


def test_every_color_in_a_layout_rule_is_a_theme_variable(rendered):
    """A hardcoded palette renders dark-on-dark under JupyterLab's dark theme.

    Only the layout rules are checked. The variable blocks above them hold the literals deliberately, as the
    last fallback for a frontend that defines neither ``--jp-*`` nor ``--pst-*``.
    """
    layout = rendered[rendered.index("/* Layout.") :]

    assert not re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", layout)


def test_the_theme_variables_are_scoped_to_the_container(rendered):
    """Defining them on ``:root`` writes document-scoped properties from inside an output cell."""
    assert ":root" not in rendered


class TestModelRepr:
    def test_a_declared_name_becomes_a_caption(self, rbc):
        assert "Marginal utility of wealth" in rbc._repr_html_()

    def test_a_symbol_renders_through_the_latex_printer(self, rbc):
        r"""A bare sympy printer would give ``mc`` as a product and ``epsilon`` as ``\epsilon``."""
        rendered = rbc._repr_html_()

        assert r"\rho_{A}" in rendered
        assert r"\varepsilon_{A,t}" in rendered

    def test_a_declared_latex_field_overrides_the_stem(self, tmp_path):
        """The symbol table and the block equations render through different calls, and both honor the override."""
        path = tmp_path / "override.gcn"
        path.write_text(
            'symbols { Y[] { latex = "\\mathcal{Y}"; }; A[] { }; alpha { }; };\n'
            "block H { identities { Y[] = alpha * A[]; A[] = 1; }; calibration { alpha = 0.3; }; };\n"
        )

        rendered = model_from_gcn(path, verbose=False)._repr_html_()
        variables, _, equations = rendered.partition("Shocks (0)")

        assert r"{\mathcal{Y}}_{t}" in variables
        assert r"{\mathcal{Y}}_{t}" in equations

    def test_a_prior_reaches_the_parameter_table(self, rbc):
        assert "Beta(alpha=" in rbc._repr_html_()

    def test_the_parameter_count_matches_the_table_under_it(self):
        """``model.params`` omits calibrated parameters and shock hyper-parameters; the table holds them."""
        model = model_from_gcn(TEST_GCNS / "open_rbc.gcn", verbose=False)

        rendered = model._repr_html_()

        assert f"Parameters ({len(model.calibration_table().rows)})" in rendered

    def test_a_value_in_scientific_notation_stays_in_math_mode(self, tmp_path):
        """``_format_value`` writes ``$...$`` for a LaTeX table, and no notebook frontend typesets that."""
        path = tmp_path / "tiny.gcn"
        path.write_text(
            "symbols { Y[] { }; A[] { }; sigma { }; };\n"
            "block H { identities { Y[] = sigma * A[]; A[] = 1; }; calibration { sigma = 1e-6; }; };\n"
        )

        rendered = model_from_gcn(path, verbose=False)._repr_html_()

        assert r"\(1 \times 10^{-6}\)" in rendered
        assert "$" not in rendered

    def test_a_definition_prints_before_the_objective_that_uses_it(self, rbc):
        """The household objective names ``u_t``, and a reader checking it against their file reads downward."""
        rendered = rbc._repr_html_()

        assert rendered.index("Definitions") < rendered.index(r"\max_{C_{t}")

    def test_a_blocks_shocks_and_calibration_reach_the_render(self, rbc):
        """Technology_Shocks exists to declare a shock, and the view claims to show the author's program."""
        rendered = rbc._repr_html_()

        assert "<p class='ge-component'>Shocks</p>" in rendered
        assert "<p class='ge-component'>Calibration</p>" in rendered
        assert "rho_A ~ maxent(Beta(), lower=0.8, upper=0.99) = 0.95" in rendered

    def test_a_block_problem_prints_as_an_optimization(self, rbc):
        """#47 asks for the program the author wrote, not headings labeled objective and controls."""
        rendered = rbc._repr_html_()

        assert r"\max_{K_{t-1}, L_{t}}" in rendered
        assert "subject to" in rendered

    def test_a_minimize_tag_flips_the_operator(self):
        """Inverting the ``is_minimize`` branch prints every cost minimization as a maximization."""
        rendered = model_from_gcn(TEST_GCNS / "rbc_2_block_minimize.gcn", verbose=False)._repr_html_()

        assert r"\min_{K_{t-1}, L_{t}}" in rendered
        assert r"\max_{K_{t-1}, L_{t}}" not in rendered

    def test_a_named_multiplier_prints_beside_its_constraint(self, rbc):
        assert r"\quad (\text{mc}_{t})" in rbc._repr_html_()

    def test_the_authored_steady_state_is_the_only_one_shown(self, rbc):
        """``steady_state_relationships`` holds the substituted form, which is unreadable beside the authored one."""
        rendered = rbc._repr_html_()

        assert rendered.count(block_heading("Steady_State")) == 1
        assert SOLVED_STEADY_STATE_TITLE not in rendered

    def test_a_caption_is_escaped(self, tmp_path):
        """A declared name is free-form prose, and an unescaped ``&`` is not valid HTML."""
        path = tmp_path / "escaped.gcn"
        path.write_text(
            'symbols { Y[] { name = "Output & income"; }; A[] { }; alpha { }; };\n'
            "block H { identities { Y[] = alpha * A[]; A[] = 1; }; calibration { alpha = 0.3; }; };\n"
        )

        rendered = model_from_gcn(path, verbose=False)._repr_html_()

        assert "Output &amp; income" in rendered
        assert "Output & income" not in rendered


def test_a_model_with_no_parsed_file_keeps_its_captions(rbc):
    """The symbol tables read declarations off the model, so only the authored block sections go missing."""
    rendered = render_model(rbc, source_ast=None)

    assert "Marginal utility of wealth" in rendered
    assert "Capital share of output" in rendered
    assert "Household" not in rendered
    assert SOLVED_STEADY_STATE_TITLE in rendered


@pytest.mark.parametrize("example", ["RBC", "New_Keynesian", "Three_Equation_NK"])
def test_every_authored_block_reaches_the_render(example):
    """A renderer that drops a block leaves a model the reader cannot check against their file."""
    model = model_from_gcn(get_example_gcn(example), verbose=False)

    rendered = model._repr_html_()

    assert "<script" not in rendered
    sections = rendered.count("<summary class='block-title'>")

    assert sections == len(model._source_ast.blocks) + SUMMARY_SECTIONS
