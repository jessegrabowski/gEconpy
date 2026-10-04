from dataclasses import asdict, dataclass, fields

import pandas as pd

from sympy.printing.latex import latex_escape

from gEconpy.model.latex import block_heading


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
class ParameterRow:
    """One parameter of a model, with everything a calibration table prints about it."""

    symbol: str
    description: str | None
    value: float | None
    prior: str | None
    source: str | None


@dataclass(frozen=True)
class CalibrationTable:
    """
    A model's parameters as table data, rendered on demand.

    The rows are built once and each renderer reads them, so LaTeX and a dataframe cannot drift apart.
    """

    rows: tuple[ParameterRow, ...] | list[ParameterRow]

    def to_frame(self) -> pd.DataFrame:
        """
        Return the table as a dataframe, one row per parameter.

        Returns
        -------
        frame : pandas.DataFrame
            Columns ``symbol``, ``description``, ``value``, ``prior`` and ``source``.
        """
        return pd.DataFrame([asdict(row) for row in self.rows], columns=[field.name for field in fields(ParameterRow)])

    def to_latex(self) -> str:
        r"""
        Render the table as a LaTeX ``tabular``.

        A column whose every entry is empty is dropped, so a model that declares no ``source`` does not print an
        empty citation column.

        Returns
        -------
        latex : str
            The tabular, ready to paste into a document.
        """
        if not self.rows:
            return ""

        # Prose columns wrap; a symbol or a number does not need to.
        columns = [
            ("Parameter", "l", lambda row: f"${row.symbol}$"),
            ("Description", r"p{0.3\linewidth}", lambda row: latex_escape(row.description or "")),
            ("Value", "l", lambda row: _format_value(row.value)),
            ("Prior", r"p{0.25\linewidth}", lambda row: latex_escape(row.prior or "")),
            ("Source", r"p{0.2\linewidth}", lambda row: latex_escape(row.source or "")),
        ]
        rendered = {heading: [render(row) for row in self.rows] for heading, _, render in columns}
        kept = [(heading, alignment) for heading, alignment, _ in columns if any(rendered[heading])]

        header = " & ".join(rf"\textbf{{{heading}}}" for heading, _ in kept)
        body = " \\\\\n".join(
            " & ".join(rendered[heading][index] for heading, _ in kept) for index in range(len(self.rows))
        )
        alignment = "".join(column for _, column in kept)
        return (
            f"\\begin{{tabular}}{{{alignment}}}\n\\hline\n{header} \\\\\n\\hline\n"
            f"{body} \\\\\n\\hline\n\\end{{tabular}}"
        )
