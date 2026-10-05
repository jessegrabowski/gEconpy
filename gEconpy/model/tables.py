from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, Literal, get_args

import pandas as pd

from sympy.printing.latex import latex_escape

from gEconpy.model.latex import block_heading

TableGroup = Literal["equations", "variables", "shocks", "parameters"]
TABLE_GROUPS: tuple[TableGroup, ...] = get_args(TableGroup)


def _format_value(value: float | None, math: tuple[str, str] = ("$", "$")) -> str:
    """
    Render a parameter value, keeping scientific notation inside math mode so it does not print literally.

    Parameters
    ----------
    value : float, optional
        The value, or None for a parameter the model solves for.
    math : tuple of (str, str), optional
        Opening and closing math delimiters for scientific notation. Defaults to a LaTeX document's ``$``,
        which an HTML renderer must override because no notebook frontend recognizes it.

    Returns
    -------
    rendered : str
        The value, empty for None.
    """
    if value is None:
        return ""

    rendered = f"{value:g}"
    if "e" not in rendered:
        return rendered

    mantissa, _, exponent = rendered.partition("e")
    opening, closing = math
    return rf"{opening}{mantissa} \times 10^{{{int(exponent)}}}{closing}"


@dataclass(frozen=True)
class _Column:
    """
    One column of a tabular table: its heading, its LaTeX alignment, and how to read it off a row.

    Parameters
    ----------
    heading : str
        Column heading, printed in bold.
    alignment : str
        LaTeX column specifier, such as ``l`` or a ``p`` box for a column of prose that should wrap.
    value : callable
        Maps a row to its already-escaped cell text.
    """

    heading: str
    alignment: str
    value: Callable[[Any], str]


def _tabular(rows: Sequence[Any], columns: Sequence[_Column]) -> str:
    r"""
    Render rows as a LaTeX ``tabular``, dropping any column that is empty for every row.

    Parameters
    ----------
    rows : sequence
        The table rows, in print order.
    columns : sequence of _Column
        The columns to consider, in print order.

    Returns
    -------
    latex : str
        The tabular, or the empty string when there are no rows.
    """
    cells = {column.heading: [column.value(row) for row in rows] for column in columns}
    kept = [column for column in columns if any(cells[column.heading])]
    if not rows:
        return ""

    header = " & ".join(rf"\textbf{{{column.heading}}}" for column in kept)
    body = " \\\\\n".join(" & ".join(cells[column.heading][index] for column in kept) for index in range(len(rows)))
    alignment = "".join(column.alignment for column in kept)
    return f"\\begin{{tabular}}{{{alignment}}}\n\\hline\n{header} \\\\\n\\hline\n{body} \\\\\n\\hline\n\\end{{tabular}}"


@dataclass(frozen=True)
class EquationRow:
    """
    One equation of a model, rendered and ready to print.

    ``block`` is the identifier as the file spells it. The renderers turn it into prose. A ``definitions``
    entry has no ``equation_id``, because the solver substitutes it away before equations are given ids.
    """

    block: str
    equation_id: str | None
    label: str | None
    left: str
    right: str


@dataclass(frozen=True)
class EquationTable:
    """
    A model's equations as table data, rendered on demand.

    The rows are built once and each renderer reads them, so LaTeX and a dataframe cannot drift apart.
    """

    rows: tuple[EquationRow, ...] | list[EquationRow]

    def to_frame(self) -> pd.DataFrame:
        """
        Return the table as a dataframe, one row per equation.

        Returns
        -------
        frame : pandas.DataFrame
            Columns ``block``, ``equation_id``, ``label`` and ``equation``, where ``equation`` is the two sides
            joined by ``=``.
        """
        return pd.DataFrame(
            [
                {
                    "block": row.block,
                    "equation_id": row.equation_id,
                    "label": row.label,
                    "equation": f"{row.left} = {row.right}",
                }
                for row in self.rows
            ],
            columns=["block", "equation_id", "label", "equation"],
        )

    def to_latex(self) -> str:
        r"""
        Render the table as a LaTeX ``align`` environment, with a heading before each block's equations.

        A row's caption becomes a ``\tag``. Headings use ``\intertext``, which is how amsmath interjects prose
        into an aligned block without breaking the alignment.

        Returns
        -------
        latex : str
            The environment, ready to paste into a document.
        """
        if not self.rows:
            return ""

        lines: list[tuple[bool, str]] = []
        heading = None
        for row in self.rows:
            if row.block != heading:
                heading = row.block
                lines.append((True, rf"\intertext{{\textbf{{{latex_escape(block_heading(heading))}}}}}"))
            tag = rf" \tag{{\text{{{latex_escape(row.label)}}}}}" if row.label else ""
            lines.append((False, rf"{row.left} &= {row.right}{tag}"))

        # A heading is not a row and takes no terminator, and one after the final row is a LaTeX error.
        last_row = max((index for index, (is_heading, _) in enumerate(lines) if not is_heading), default=-1)
        body = "\n".join(
            text if is_heading or index == last_row else text + r" \\" for index, (is_heading, text) in enumerate(lines)
        )
        return f"\\begin{{align}}\n{body}\n\\end{{align}}"


@dataclass(frozen=True)
class _TabularTable(ABC):
    """
    Shared behavior for a table that prints as a LaTeX ``tabular``.

    A subclass supplies its row type and its columns. The rows are built once and each renderer reads them,
    so LaTeX and a dataframe cannot drift apart.
    """

    rows: Sequence[Any]

    _row_type: ClassVar[type]

    @abstractmethod
    def _columns(self) -> list[_Column]:
        """Return the columns this table prints, in order."""

    def to_frame(self) -> pd.DataFrame:
        """
        Return the table as a dataframe, one row per entry.

        Returns
        -------
        frame : pandas.DataFrame
            One column per field of the row type, in declaration order.
        """
        return pd.DataFrame(
            [asdict(row) for row in self.rows], columns=[field.name for field in fields(self._row_type)]
        )

    def to_latex(self) -> str:
        r"""
        Render the table as a LaTeX ``tabular``, dropping any column that is empty for every row.

        Returns
        -------
        latex : str
            The tabular, ready to paste into a document.
        """
        return _tabular(self.rows, self._columns())


@dataclass(frozen=True)
class SymbolRow:
    """One variable or shock of a model, with the caption its ``symbols`` entry declares."""

    symbol: str
    description: str | None


@dataclass(frozen=True)
class SymbolTable(_TabularTable):
    """A model's variables or shocks as table data, rendered on demand."""

    rows: tuple[SymbolRow, ...] | list[SymbolRow]

    _row_type: ClassVar[type] = SymbolRow

    def _columns(self) -> list[_Column]:
        return [
            _Column("Symbol", "l", lambda row: f"${row.symbol}$"),
            _Column("Description", r"p{0.5\linewidth}", lambda row: latex_escape(row.description or "")),
        ]


@dataclass(frozen=True)
class ParameterRow:
    """One parameter of a model, with everything a calibration table prints about it."""

    symbol: str
    description: str | None
    value: float | None
    prior: str | None
    source: str | None


@dataclass(frozen=True)
class CalibrationTable(_TabularTable):
    """A model's parameters as table data, rendered on demand."""

    rows: tuple[ParameterRow, ...] | list[ParameterRow]

    _row_type: ClassVar[type] = ParameterRow

    def _columns(self) -> list[_Column]:
        # Prose columns wrap; a symbol or a number does not need to.
        return [
            _Column("Parameter", "l", lambda row: f"${row.symbol}$"),
            _Column("Description", r"p{0.3\linewidth}", lambda row: latex_escape(row.description or "")),
            _Column("Value", "l", lambda row: _format_value(row.value)),
            _Column("Prior", r"p{0.25\linewidth}", lambda row: latex_escape(row.prior or "")),
            _Column("Source", r"p{0.2\linewidth}", lambda row: latex_escape(row.source or "")),
        ]
