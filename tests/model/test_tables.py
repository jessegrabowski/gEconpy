import shutil
import subprocess

import preliz
import pytest

from gEconpy import model_from_gcn
from gEconpy.data import get_example_gcn
from gEconpy.model.tables import (
    PRIOR_STAT_HEADINGS,
    PRIOR_STATS,
    TABLE_GROUPS,
    TABLE_WRITERS,
    CalibrationTable,
    EquationRow,
    EquationTable,
    ParameterRow,
    SymbolTable,
    _format_value,
)
from tests.conftest import TEST_GCNS


@pytest.fixture(scope="module")
def rbc():
    """Built once: every test here reads the same table, and building the model dominates their runtime."""
    return model_from_gcn(get_example_gcn("RBC"), verbose=False)


class TestEquationTable:
    def test_a_row_carries_its_block_id_control_and_both_sides(self, rbc):
        row = next(r for r in rbc.table("equations").rows if r.equation_id == "Household.foc.C")

        assert row.block == "Household"
        assert row.label is None
        assert row.foc_control == "C_{t}"
        assert row.right == "0"
        assert r"\lambda_{t}" in row.left

    def test_a_definition_row_has_no_equation_id(self, rbc):
        """The solver substitutes definitions away before equations are given ids."""
        definitions = [r for r in rbc.table("equations").rows if r.equation_id is None]

        assert [r.left for r in definitions] == ["u_{t}"]
        assert all(r.block == "Household" for r in definitions)

    def test_every_equation_gets_a_row(self, rbc):
        rows = rbc.table("equations").rows

        assert [r.equation_id for r in rows if r.equation_id is not None] == rbc._equation_ids

    @pytest.mark.parametrize(
        "example, expected",
        [
            ("RBC", ["Household", "Firm", "Technology_Shocks"]),
            ("Baxter_King_1993", ["Household", "Firm", "Fiscal_Authority"]),
        ],
        ids=["definitions_in_the_first_block", "definitions_in_a_later_block"],
    )
    def test_rows_stay_in_source_block_order(self, example, expected):
        """
        A block appears once, where the file puts it.

        RBC alone cannot pin this: its definitions are in its first block, so emitting all definitions ahead of
        all equations happens to produce the right order. Baxter-King has them in a later block.
        """
        blocks = [r.block for r in model_from_gcn(get_example_gcn(example), verbose=False).table("equations").rows]

        assert list(dict.fromkeys(blocks)) == expected
        assert blocks == sorted(blocks, key=expected.index)


class TestRendering:
    @pytest.mark.parametrize(
        "example, expected",
        [
            ("RBC", ["Household", "Firm", "Technology Shocks"]),
            ("Baxter_King_1993", ["Household", "Firm", "Fiscal Authority"]),
        ],
        ids=["definitions_in_the_first_block", "definitions_in_a_later_block"],
    )
    def test_each_block_gets_exactly_one_heading(self, example, expected):
        """A repeated heading means the rows were grouped by something other than the block."""
        rendered = model_from_gcn(get_example_gcn(example), verbose=False).table("equations").to_latex()

        headings = [line for line in rendered.splitlines() if line.startswith(r"\intertext")]
        assert headings == [rf"\intertext{{\textbf{{{name}}}}}" for name in expected]

    def test_a_row_keeps_the_block_identifier_the_file_spells(self, rbc):
        """The renderers turn it into prose, so the row has to keep the name a caller could filter on."""
        assert {r.block for r in rbc.table("equations").rows} == {"Household", "Firm", "Technology_Shocks"}

    def test_only_the_last_equation_line_lacks_a_terminator(self, rbc):
        r"""A stray terminator before \\end{align} is a LaTeX error, and one on a heading breaks the alignment."""
        lines = rbc.table("equations").to_latex().splitlines()[1:-1]

        assert all(not line.endswith(r"\\") for line in lines if line.startswith(r"\intertext"))
        equations = [line for line in lines if not line.startswith(r"\intertext")]
        assert all(line.endswith(r"\\") for line in equations[:-1])
        assert not equations[-1].endswith(r"\\")

    def test_a_block_heading_is_escaped(self):
        table = EquationTable([EquationRow(block="R&D Lab", equation_id=None, label=None, left="x", right="1")])

        assert r"\textbf{R\&D Lab}" in table.to_latex()

    def test_the_frame_has_one_row_per_equation_with_the_sides_joined(self, rbc):
        frame = rbc.table("equations").to_frame()

        assert list(frame.columns) == ["block", "equation_id", "label", "equation"]
        assert len(frame) == len(rbc.table("equations").rows)
        assert frame.loc[frame.equation_id == "Household.foc.C", "equation"].item().endswith(" = 0")

    def test_the_frame_and_the_latex_hold_the_same_rows(self, rbc):
        """One builder feeds both, so a row cannot appear in one rendering and not the other."""
        table = rbc.table("equations")
        equations = [line for line in table.to_latex().splitlines()[1:-1] if not line.startswith(r"\intertext")]

        assert len(equations) == len(table.rows) == len(table.to_frame())
        assert all(row.left in line for row, line in zip(table.rows, equations, strict=True))


class TestCalibrationTable:
    def test_a_row_carries_the_symbol_description_value_and_prior(self, rbc):
        row = next(r for r in rbc.table("parameters").rows if r.symbol == r"\alpha")

        assert row.description == "Capital share of output"
        assert row.value == pytest.approx(0.35)
        assert row.prior_family == "Beta"
        assert row.prior_stat("mean") == pytest.approx(0.344, abs=1e-3)

    def test_a_parameter_renders_by_the_same_rules_as_a_variable(self, tmp_path):
        r"""
        A parameter is a plain Symbol, so sympy's default printer would reach it instead.

        Every parameter in RBC renders identically either way, which is why this builds a model with two that
        do not: sympy gives ``mc`` and ``\\epsilon``.
        """
        path = tmp_path / "params.gcn"
        path.write_text(
            "symbols { Y[] { }; A[] { }; };\n"
            "block H { identities { Y[] = mc * A[] + epsilon; A[] = 1; };"
            " calibration { mc = 1.0; epsilon = 0.1; }; };\n"
        )

        symbols = {r.symbol for r in model_from_gcn(path, verbose=False).table("parameters").rows}

        assert r"\text{mc}" in symbols
        assert r"\varepsilon" in symbols

    def test_a_calibrated_parameter_has_no_value(self):
        """It is solved for rather than set, so printing a number would invent one."""
        model = model_from_gcn(TEST_GCNS / "one_block_2_no_extra.gcn", verbose=False)

        calibrated = next(r for r in model.table("parameters").rows if r.symbol == r"\alpha")

        assert calibrated.value is None
        assert [r.symbol for r in model.table("parameters").rows][-1] == r"\alpha"

    def test_an_empty_column_is_dropped(self):
        """Built here rather than from a fixture, so adding a source to a shipped model cannot silently break it."""
        table = CalibrationTable(
            [
                ParameterRow(symbol=r"\alpha", description="Capital share", value=0.35, prior=None, source=None),
                ParameterRow(symbol=r"\beta", description="Discount factor", value=0.99, prior=None, source=None),
            ]
        )

        rendered = table.to_latex()

        assert "Source" not in rendered
        assert "Prior" not in rendered
        assert "Description" in rendered

    def test_a_declared_source_brings_its_column_back(self, tmp_path):
        path = tmp_path / "sourced.gcn"
        path.write_text(
            'symbols { Y[] { }; A[] { }; alpha { name = "Capital share"; source = "Smets & Wouters (2007)"; }; };\n'
            "block H { identities { Y[] = alpha * A[]; A[] = 1; }; calibration { alpha = 0.3; }; };\n"
        )

        rendered = model_from_gcn(path, verbose=False).table("parameters").to_latex()

        assert "Source" in rendered
        assert r"Smets \& Wouters (2007)" in rendered

    def test_the_frame_keeps_every_column(self, rbc):
        """to_latex drops an empty column for printing; the frame is data and keeps it."""
        frame = rbc.table("parameters").to_frame()

        assert list(frame.columns) == [
            "symbol",
            "description",
            "value",
            "prior",
            "prior_mean",
            "prior_std",
            "prior_parameters",
            "source",
        ]
        assert len(frame) == len(rbc.table("parameters").rows)


class TestEmptyTable:
    """An empty table must not emit a tabular with no columns or an align with no rows; pdflatex rejects both."""

    @pytest.mark.parametrize("table", [EquationTable([]), CalibrationTable([])], ids=["equations", "calibration"])
    def test_an_empty_table_renders_as_nothing(self, table):
        assert table.to_latex() == ""

    @pytest.mark.parametrize("style", [{}, {"caption": "Shocks"}, {"label": "tab:s"}, {"size": "small"}])
    def test_an_empty_table_stays_empty_under_any_style(self, style):
        """A caption on no table still floats, numbering a phantom entry in the list of tables."""
        assert SymbolTable([]).to_latex(**style) == ""

    @pytest.mark.parametrize(
        "table, columns",
        [
            (EquationTable([]), ["block", "equation_id", "label", "equation"]),
            (
                CalibrationTable([]),
                ["symbol", "description", "value", "prior", "prior_mean", "prior_std", "prior_parameters", "source"],
            ),
        ],
        ids=["equations", "calibration"],
    )
    def test_an_empty_frame_keeps_its_columns(self, table, columns):
        """Returning a frame with no columns breaks the contract the docstring states."""
        assert list(table.to_frame().columns) == columns


class TestValueFormatting:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.35, "0.35"),
            (2.0, "2"),
            (1e-07, r"$1 \times 10^{-7}$"),
            (1.23457e08, r"$1.23457 \times 10^{8}$"),
            (None, ""),
        ],
        ids=["decimal", "integral", "small", "large", "calibrated"],
    )
    def test_a_value_never_prints_bare_scientific_notation(self, value, expected):
        """``1e-07`` in a tabular cell is outside math mode and prints literally."""
        assert _format_value(value) == expected


def test_the_rendered_calibration_has_the_expected_shape():
    """Pins column order, the hline structure and the escaping, which field assertions leave unchecked.

    The rows are written out rather than taken from a model, because a prior's parameters come from a
    moment-matching solve and their third significant figure moves with the scipy build.
    """
    table = CalibrationTable(
        rows=[
            ParameterRow(
                symbol=r"\alpha",
                description="Capital share of output",
                value=0.35,
                prior=preliz.Beta(alpha=21.8, beta=41.6),
                source="Smets & Wouters (2007)",
            ),
            ParameterRow(symbol=r"\delta", description="Depreciation rate", value=0.02, prior=None, source=None),
        ]
    )

    assert table.to_latex() == (
        "\\begin{tabular}{lp{0.3\\linewidth}llrrp{0.2\\linewidth}}\n"
        "\\hline\n"
        "\\textbf{Parameter} & \\textbf{Description} & \\textbf{Value} & \\textbf{Prior} & "
        "\\textbf{Mean} & \\textbf{S.D.} & \\textbf{Source} \\\\\n"
        "\\hline\n"
        "$\\alpha$ & Capital share of output & 0.35 & Beta & 0.344 & 0.0592 & Smets \\& Wouters (2007) \\\\\n"
        "$\\delta$ & Depreciation rate & 0.02 &  &  &  &  \\\\\n"
        "\\hline\n"
        "\\end{tabular}"
    )


@pytest.mark.parametrize("group", ["equations", "parameters"])
def test_the_rendered_table_compiles(rbc, group, tmp_path):
    r"""
    The feature's whole contract is valid LaTeX, and an empty table once produced output pdflatex rejected.

    ``amssymb`` is needed as well as ``amsmath``: the expectation operator renders as ``\\mathbb{E}``.
    """
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        pytest.skip("pdflatex not installed")

    body = rbc.table(group).to_latex()
    source = tmp_path / "table.tex"
    source.write_text(
        f"\\documentclass{{article}}\n\\usepackage{{amsmath,amssymb}}\n\\begin{{document}}\n{body}\n\\end{{document}}\n"
    )

    result = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", source.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout[-2000:]


@pytest.mark.parametrize(
    "group, columns",
    [
        ("equations", ["block", "equation_id", "label", "equation"]),
        ("variables", ["symbol", "description"]),
        ("shocks", ["symbol", "description"]),
        (
            "parameters",
            ["symbol", "description", "value", "prior", "prior_mean", "prior_std", "prior_parameters", "source"],
        ),
    ],
)
def test_every_group_declares_the_columns_its_frame_carries(rbc, group, columns):
    table = rbc.table(group)

    assert list(table.to_frame().columns) == columns
    assert len(table.to_frame()) == len(table.rows)


@pytest.mark.parametrize("group", ["variables", "shocks"])
def test_a_symbol_group_covers_every_symbol(rbc, group):
    """A row builder that filtered or deduplicated would leave a symbol out of the paper."""
    assert len(rbc.table(group).rows) == len(getattr(rbc, group))


def test_the_equations_group_covers_every_equation(rbc):
    """Definitions have no id and are extra rows; every solved equation must still appear, in order."""
    ids = [row.equation_id for row in rbc.table("equations").rows if row.equation_id is not None]

    assert ids == rbc._equation_ids


def test_an_unknown_group_raises(rbc):
    """A typo in a group name should fail loudly rather than return an empty table."""
    with pytest.raises(ValueError, match="group must be one of equations, variables, shocks, parameters"):
        rbc.table("parameter")


def test_a_symbol_row_carries_its_declared_caption_and_latex(rbc):
    """The variables group reads the same declarations the notebook repr does."""
    frame = rbc.table("variables").to_frame().set_index("symbol")

    assert frame.loc[r"\lambda_{t}", "description"] == "Marginal utility of wealth"
    assert frame.loc["K_{t}", "description"] == "Capital stock"


def test_a_shock_row_renders_through_the_latex_printer(rbc):
    r"""``epsilon`` renders as ``\\varepsilon``, which a bare sympy printer would not do."""
    symbols = rbc.table("shocks").to_frame()["symbol"].tolist()

    assert symbols == [r"\varepsilon_{A,t}"]


def test_a_symbol_table_drops_the_description_column_when_nothing_declares_one(tmp_path):
    """A model with no declared names should not print an empty caption column."""
    path = tmp_path / "bare.gcn"
    path.write_text(
        "symbols { Y[] { }; A[] { }; alpha { }; };\n"
        "block H { identities { Y[] = alpha * A[]; A[] = 1; }; calibration { alpha = 0.3; }; };\n"
    )

    rendered = model_from_gcn(path, verbose=False).table("variables").to_latex()

    assert "Description" not in rendered
    assert "Symbol" in rendered


class TestWriteTable:
    def test_an_unknown_writer_raises(self, rbc):
        with pytest.raises(ValueError, match="writer must be one of latex"):
            rbc.write_table("parameters", writer="typst")

    def test_a_caption_and_label_wrap_the_table(self, rbc):
        """Neither can be referenced outside a table environment, so asking for one implies the wrapper."""
        rendered = rbc.write_table("parameters", caption="Calibrated parameters", label="tab:calib")

        assert r"\begin{table}[htbp]" in rendered
        assert r"\caption{Calibrated parameters}" in rendered
        assert r"\label{tab:calib}" in rendered

    def test_a_caption_carries_math_unescaped(self, rbc):
        """A table caption in this literature routinely carries math, so it is the author's own LaTeX."""
        rendered = rbc.write_table("parameters", caption=r"Calibration of $\beta$")

        assert r"\caption{Calibration of $\beta$}" in rendered

    def test_no_style_leaves_a_bare_tabular(self, rbc):
        rendered = rbc.write_table("parameters")

        assert r"\begin{table}" not in rendered
        assert rendered.startswith(r"\begin{tabular}")

    def test_booktabs_replaces_the_hlines(self, rbc):
        rendered = rbc.write_table("variables", booktabs=True)

        assert r"\toprule" in rendered
        assert r"\midrule" in rendered
        assert r"\bottomrule" in rendered
        assert r"\hline" not in rendered

    def test_size_is_scoped_to_the_table(self, rbc):
        """A size command is a declaration, so unbraced it shrinks every paragraph after the table."""
        rendered = rbc.write_table("variables", size="footnotesize")

        assert rendered.startswith("{\\footnotesize\n")
        assert rendered.endswith("}")

    def test_widths_replace_the_column_spec(self, rbc):
        rendered = rbc.write_table("variables", widths=["c", r"p{0.9\linewidth}"])

        assert r"\begin{tabular}{cp{0.9\linewidth}}" in rendered

    def test_widths_of_the_wrong_length_raise(self, rbc):
        """A dropped empty column changes how many specifiers are needed, so the count is checked."""
        with pytest.raises(ValueError, match="one specifier per printed column"):
            rbc.write_table("variables", widths=["c"])

    def test_a_path_receives_the_markup(self, rbc, tmp_path):
        out = tmp_path / "table.tex"

        returned = rbc.write_table("shocks", path=out)

        assert out.read_text() == returned

    def test_a_fully_styled_table_compiles(self, rbc, tmp_path):
        """Every style option at once is the combination most likely to produce invalid LaTeX."""
        pdflatex = shutil.which("pdflatex")
        if pdflatex is None:
            pytest.skip("pdflatex not installed")

        body = rbc.write_table(
            "parameters",
            caption="Calibration",
            label="tab:calib",
            size="small",
            booktabs=True,
        )
        source = tmp_path / "styled.tex"
        source.write_text(
            "\\documentclass{article}\n\\usepackage{amsmath,amssymb,booktabs}\n"
            f"\\begin{{document}}\n{body}\n\\end{{document}}\n"
        )

        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", source.name],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stdout[-2000:]


@pytest.mark.parametrize("group", TABLE_GROUPS)
@pytest.mark.parametrize("writer", list(TABLE_WRITERS))
def test_every_announced_writer_renders_every_group(rbc, writer, group):
    """A writer wired onto one table class but not the others passes a single-group test and fails in use."""
    assert rbc.write_table(group, writer=writer)


def test_a_style_option_the_group_does_not_take_raises(rbc):
    """Equations render as an align environment, which has no rules to style, so booktabs must not be ignored."""
    with pytest.raises(TypeError, match="'equations' takes no style option 'booktabs'"):
        rbc.write_table("equations", booktabs=True)


def test_a_build_option_the_group_does_not_take_raises(rbc):
    """``expectations`` means nothing outside the equations group, and silence would hide a flag going nowhere."""
    with pytest.raises(TypeError, match="'variables' takes no build option 'expectations'"):
        rbc.table("variables", expectations=False)


def test_an_unknown_style_option_names_the_group_not_the_renderer(rbc):
    """Python's own message names the bound method, which is private, so a typo points at the wrong thing."""
    with pytest.raises(TypeError, match="Accepted: caption, label, size, widths, booktabs"):
        rbc.write_table("parameters", captoin="Calibration")


def test_an_empty_table_rejects_widths():
    """An empty table prints no columns, so any width count is wrong, and silence would hide a bad script."""
    with pytest.raises(ValueError, match="one specifier per printed column"):
        SymbolTable([]).to_latex(widths=["c", "c"])


@pytest.mark.parametrize(
    "options, headings",
    [
        ({}, ["Prior", "Mean", "S.D."]),
        ({"prior_stats": ["median", "skewness", "kurtosis"]}, ["Prior", "Median", "Skew", "Kurtosis"]),
        ({"prior_stats": [], "include_prior_params": True}, ["Prior", "Parameters"]),
        ({"include_prior_params": True}, ["Prior", "Mean", "S.D.", "Parameters"]),
    ],
    ids=["default", "other_stats", "params_only", "stats_and_params"],
)
def test_the_prior_presentation_is_chosen_by_the_caller(rbc, options, headings):
    """A prior as one string is the least readable column in the table, so the caller picks its columns."""
    rendered = rbc.table("parameters", **options).to_latex()
    header = next(line for line in rendered.splitlines() if "textbf{Parameter}" in line)
    printed = [
        h for h in ["Prior", "Mean", "S.D.", "Median", "Skew", "Kurtosis", "Parameters"] if f"\\textbf{{{h}}}" in header
    ]

    assert printed == headings


def test_an_unknown_prior_statistic_raises(rbc):
    """A typo must not produce a silently empty column in a table bound for a paper."""
    with pytest.raises(ValueError, match="got medain"):
        rbc.table("parameters", prior_stats=["medain"])


def test_an_unknown_statistic_raises_even_when_no_parameter_has_a_prior():
    """Validating against the rows would pass a typo on a model whose parameters are all calibrated."""
    with pytest.raises(ValueError, match="got medain"):
        CalibrationTable(rows=[ParameterRow("x", None, 1.0, None, None)], prior_stats=["medain"])


def test_a_distribution_method_that_is_not_a_statistic_is_rejected():
    """``rvs`` and ``plot_pdf`` are on the base class too, so presence alone is not the test."""
    with pytest.raises(ValueError, match="got rvs"):
        CalibrationTable(rows=[], prior_stats=["rvs"])


@pytest.mark.parametrize("name", PRIOR_STATS)
def test_every_announced_statistic_renders(rbc, name):
    """A name in PRIOR_STATS that some family cannot compute would raise only for that family's priors."""
    heading = PRIOR_STAT_HEADINGS.get(name, name.title())

    frame = rbc.table("parameters", prior_stats=[name]).to_frame()
    rendered = rbc.table("parameters", prior_stats=[name]).to_latex()

    assert rf"\textbf{{{heading}}}" in rendered
    assert frame[f"prior_{name}"].notna().all()


def test_a_prior_prints_its_moments_to_three_figures(rbc):
    """``%g`` gives six, and a prior mean of 0.343849 is noise in a paper."""
    rendered = rbc.table("parameters").to_latex()

    assert "& 0.344 &" in rendered
    assert "0.343849" not in rendered


def test_a_parameter_with_no_prior_reports_nothing_about_one():
    """A calibrated parameter has a value and no distribution, and must not print a family or a moment."""
    row = ParameterRow(symbol=r"\delta", description="Depreciation rate", value=0.02, prior=None, source=None)

    assert row.prior_family == ""
    assert row.prior_stat("mean") is None
    assert row.prior_parameters == ""


def test_a_table_of_calibrated_parameters_drops_every_prior_column():
    """The empty-column rule should take the family and the moments out together, not leave blank headings."""
    table = CalibrationTable(
        rows=[ParameterRow(symbol=r"\delta", description="Depreciation rate", value=0.02, prior=None, source=None)],
        include_prior_params=True,
    )

    rendered = table.to_latex()

    assert not [h for h in ["Prior", "Mean", "S.D.", "Parameters"] if f"\\textbf{{{h}}}" in rendered]
    assert r"\textbf{Value}" in rendered


def test_prior_parameters_rounds_like_the_moment_columns_beside_it():
    """The raw fitted values run to six figures, which is the noise this column exists to avoid."""
    row = ParameterRow(
        symbol=r"\alpha", description=None, value=None, prior=preliz.Beta(alpha=21.8346, beta=41.6247), source=None
    )

    assert row.prior_parameters == "alpha=21.8, beta=41.6"
