import shutil
import subprocess

import pytest

from gEconpy import model_from_gcn
from gEconpy.data import get_example_gcn
from gEconpy.model.tables import (
    CalibrationTable,
    EquationRow,
    EquationTable,
    ParameterRow,
    _format_value,
)
from tests.conftest import TEST_GCNS


@pytest.fixture(scope="module")
def rbc():
    """Built once: every test here reads the same table, and building the model dominates their runtime."""
    return model_from_gcn(get_example_gcn("RBC"), verbose=False)


class TestEquationTable:
    def test_a_row_carries_its_block_id_caption_and_both_sides(self, rbc):
        row = next(r for r in rbc.equation_table().rows if r.equation_id == "Household.foc.C")

        assert row.block == "Household"
        assert row.label == "Household first-order condition for Consumption"
        assert row.right == "0"
        assert r"\lambda_{t}" in row.left

    def test_a_definition_row_has_no_equation_id(self, rbc):
        """The solver substitutes definitions away before equations are given ids."""
        definitions = [r for r in rbc.equation_table().rows if r.equation_id is None]

        assert [r.left for r in definitions] == ["u_{t}"]
        assert all(r.block == "Household" for r in definitions)

    def test_every_equation_gets_a_row(self, rbc):
        rows = rbc.equation_table().rows

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
        blocks = [r.block for r in model_from_gcn(get_example_gcn(example), verbose=False).equation_table().rows]

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
        rendered = model_from_gcn(get_example_gcn(example), verbose=False).equation_table().to_latex()

        headings = [line for line in rendered.splitlines() if line.startswith(r"\intertext")]
        assert headings == [rf"\intertext{{\textbf{{{name}}}}}" for name in expected]

    def test_a_row_keeps_the_block_identifier_the_file_spells(self, rbc):
        """The renderers turn it into prose, so the row has to keep the name a caller could filter on."""
        assert {r.block for r in rbc.equation_table().rows} == {"Household", "Firm", "Technology_Shocks"}

    def test_only_the_last_equation_line_lacks_a_terminator(self, rbc):
        r"""A stray terminator before \\end{align} is a LaTeX error, and one on a heading breaks the alignment."""
        lines = rbc.equation_table().to_latex().splitlines()[1:-1]

        assert all(not line.endswith(r"\\") for line in lines if line.startswith(r"\intertext"))
        equations = [line for line in lines if not line.startswith(r"\intertext")]
        assert all(line.endswith(r"\\") for line in equations[:-1])
        assert not equations[-1].endswith(r"\\")

    def test_a_block_heading_is_escaped(self):
        table = EquationTable([EquationRow(block="R&D Lab", equation_id=None, label=None, left="x", right="1")])

        assert r"\textbf{R\&D Lab}" in table.to_latex()

    def test_the_frame_has_one_row_per_equation_with_the_sides_joined(self, rbc):
        frame = rbc.equation_table().to_frame()

        assert list(frame.columns) == ["block", "equation_id", "label", "equation"]
        assert len(frame) == len(rbc.equation_table().rows)
        assert frame.loc[frame.equation_id == "Household.foc.C", "equation"].item().endswith(" = 0")

    def test_the_frame_and_the_latex_hold_the_same_rows(self, rbc):
        """One builder feeds both, so a row cannot appear in one rendering and not the other."""
        table = rbc.equation_table()
        equations = [line for line in table.to_latex().splitlines()[1:-1] if not line.startswith(r"\intertext")]

        assert len(equations) == len(table.rows) == len(table.to_frame())
        assert all(row.left in line for row, line in zip(table.rows, equations, strict=True))


class TestCalibrationTable:
    def test_a_row_carries_the_symbol_description_value_and_prior(self, rbc):
        row = next(r for r in rbc.calibration_table().rows if r.symbol == r"\alpha")

        assert row.description == "Capital share of output"
        assert row.value == pytest.approx(0.35)
        assert row.prior.startswith("Beta(")

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

        symbols = {r.symbol for r in model_from_gcn(path, verbose=False).calibration_table().rows}

        assert r"\text{mc}" in symbols
        assert r"\varepsilon" in symbols

    def test_a_calibrated_parameter_has_no_value(self):
        """It is solved for rather than set, so printing a number would invent one."""
        model = model_from_gcn(TEST_GCNS / "one_block_2_no_extra.gcn", verbose=False)

        calibrated = next(r for r in model.calibration_table().rows if r.symbol == r"\alpha")

        assert calibrated.value is None
        assert [r.symbol for r in model.calibration_table().rows][-1] == r"\alpha"

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

        rendered = model_from_gcn(path, verbose=False).calibration_table().to_latex()

        assert "Source" in rendered
        assert r"Smets \& Wouters (2007)" in rendered

    def test_the_frame_keeps_every_column(self, rbc):
        """to_latex drops an empty column for printing; the frame is data and keeps it."""
        frame = rbc.calibration_table().to_frame()

        assert list(frame.columns) == ["symbol", "description", "value", "prior", "source"]
        assert len(frame) == len(rbc.calibration_table().rows)


class TestEmptyTable:
    """An empty table must not emit a tabular with no columns or an align with no rows; pdflatex rejects both."""

    @pytest.mark.parametrize("table", [EquationTable([]), CalibrationTable([])], ids=["equations", "calibration"])
    def test_an_empty_table_renders_as_nothing(self, table):
        assert table.to_latex() == ""

    @pytest.mark.parametrize(
        "table, columns",
        [
            (EquationTable([]), ["block", "equation_id", "label", "equation"]),
            (CalibrationTable([]), ["symbol", "description", "value", "prior", "source"]),
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
                prior="Beta(alpha=21.8, beta=41.6)",
                source="Smets & Wouters (2007)",
            ),
            ParameterRow(symbol=r"\delta", description="Depreciation rate", value=0.02, prior=None, source=None),
        ]
    )

    assert table.to_latex() == (
        "\\begin{tabular}{lp{0.3\\linewidth}lp{0.25\\linewidth}p{0.2\\linewidth}}\n"
        "\\hline\n"
        "\\textbf{Parameter} & \\textbf{Description} & \\textbf{Value} & \\textbf{Prior} & \\textbf{Source} \\\\\n"
        "\\hline\n"
        "$\\alpha$ & Capital share of output & 0.35 & Beta(alpha=21.8, beta=41.6) & Smets \\& Wouters (2007) \\\\\n"
        "$\\delta$ & Depreciation rate & 0.02 &  &  \\\\\n"
        "\\hline\n"
        "\\end{tabular}"
    )


@pytest.mark.parametrize("renderer", ["equation_table", "calibration_table"])
def test_the_rendered_table_compiles(rbc, renderer, tmp_path):
    r"""
    The feature's whole contract is valid LaTeX, and an empty table once produced output pdflatex rejected.

    ``amssymb`` is needed as well as ``amsmath``: the expectation operator renders as ``\\mathbb{E}``.
    """
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        pytest.skip("pdflatex not installed")

    body = getattr(rbc, renderer)().to_latex()
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
