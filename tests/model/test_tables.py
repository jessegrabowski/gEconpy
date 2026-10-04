import pytest

from gEconpy import model_from_gcn
from gEconpy.data import get_example_gcn
from gEconpy.model.tables import EquationRow, EquationTable


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
