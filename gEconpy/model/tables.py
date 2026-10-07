import textwrap

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, Literal, get_args

import pandas as pd

from preliz.distributions.distributions import Distribution
from sympy.printing.latex import latex_escape

from gEconpy.classes.time_aware_symbol import (
    render_latex,
    render_name_latex,
    render_name_typst,
    render_typst,
)
from gEconpy.model.latex import block_heading, definition_rows, equation_sides_latex
from gEconpy.model.typst import definition_rows_typst, equation_sides_typst

TableGroup = Literal["equations", "variables", "shocks", "parameters"]
Markup = Literal["latex", "typst"]
# The statistics every preliz family reports, which is the vocabulary a caller may ask a prior for. A name
# outside it is rejected rather than rendered empty, because an empty column in a paper is not a visible error.
PRIOR_STATS = ("mean", "median", "mode", "std", "var", "skewness", "kurtosis", "entropy")
# Headings for the names a title case would get wrong.
PRIOR_STAT_HEADINGS = {"std": "S.D.", "var": "Variance", "skewness": "Skew", "kurtosis": "Kurtosis"}
DEFAULT_PRIOR_STATS = ("mean", "std")
# A prior mean of 0.343849 is noise in a paper, and three figures is what the literature prints.
PRIOR_STAT_PRECISION = 3
TABLE_GROUPS: tuple[TableGroup, ...] = get_args(TableGroup)
# Writer name to the method that renders it. The name is also the markup the rows are built in, because a
# table's cells carry rendered markup rather than model objects. ``test_every_announced_writer_renders_every_group``
# parametrizes over this mapping and every group, so a writer added here without a method on all three tables fails.
TABLE_WRITERS: dict[Markup, str] = {"latex": "to_latex", "typst": "to_typst"}


def _format_value(
    value: float | None,
    math: tuple[str, str] = ("$", "$"),
    precision: int = 6,
    times: str = r"\times",
) -> str:
    r"""
    Render a parameter value, keeping scientific notation inside math mode so it does not print literally.

    Parameters
    ----------
    value : float, optional
        The value, or None for a parameter the model solves for.
    math : tuple of (str, str), optional
        Opening and closing math delimiters for scientific notation. Defaults to a LaTeX document's ``$``,
        which an HTML renderer must override because no notebook frontend recognizes it.
    precision : int, optional
        Significant figures. Defaults to 6, which is what ``%g`` gives.
    times : str, optional
        The multiplication sign for scientific notation. Defaults to LaTeX's ``\times``.

    Returns
    -------
    rendered : str
        The value, empty for None.
    """
    if value is None:
        return ""

    rendered = f"{value:.{precision}g}"
    if "e" not in rendered:
        return rendered

    mantissa, _, exponent = rendered.partition("e")
    opening, closing = math
    return rf"{opening}{mantissa} {times} 10^{{{int(exponent)}}}{closing}"


def _rules(booktabs: bool) -> tuple[str, str, str]:
    """Return the top, middle and bottom rule commands, in booktabs or plain LaTeX."""
    if booktabs:
        return r"\toprule", r"\midrule", r"\bottomrule"
    return r"\hline", r"\hline", r"\hline"


def _wrap(
    body: str,
    caption: str | None,
    label: str | None,
    size: str | None,
) -> str:
    r"""
    Wrap a rendered body in the LaTeX a document needs around it.

    A caption or a label puts the body in a ``table`` environment, because neither can be referenced outside
    one. A size is a declaration, so it applies until the enclosing group ends.

    Parameters
    ----------
    body : str
        The rendered tabular or environment. An empty body wraps to nothing, because a caption on no table
        still floats.
    caption : str, optional
        Caption text, passed through as LaTeX so it can carry math. Defaults to no caption.
    label : str, optional
        Label for ``\ref``. Defaults to no label.
    size : str, optional
        A LaTeX size command without its backslash, such as ``small``. Defaults to the document's size.

    Returns
    -------
    latex : str
        The wrapped body.
    """
    # A caption on nothing still floats, numbering a phantom entry in the list of tables.
    if not body:
        return ""
    if size:
        # Braced, because a size command is a declaration that would otherwise run to the end of whatever
        # group the caller pasted the table into.
        body = f"{{\\{size}\n{body}\n}}"
    if caption is None and label is None:
        return body

    lines = ["\\begin{table}[htbp]", "\\centering", body]
    if caption is not None:
        # Passed through as LaTeX, because a table caption in this literature routinely carries math.
        lines.append(rf"\caption{{{caption}}}")
    if label is not None:
        lines.append(rf"\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


@dataclass(frozen=True)
class _Dialect:
    """
    Everything that differs between two markup languages, gathered in one place.

    A column says what a cell *is* and the dialect says how to write it, which is what lets one set of column
    definitions render in two languages. The symbol and equation renderers are here for the same reason: a
    table's cells hold rendered markup, so the language is chosen when the rows are built.

    Parameters
    ----------
    escape : callable
        Makes a string safe to print as text in this language.
    math : callable
        Wraps already-rendered math markup in this language's inline math delimiters.
    times : str
        This language's multiplication sign, for a value in scientific notation.
    column : callable
        Maps a column to its specifier, taking the column and returning the language's own spelling.
    render_symbol : callable
        Renders one time-aware symbol, taking an optional declared stem.
    render_name : callable
        Renders a parameter's name, taking an optional declared stem.
    equation_sides : callable
        Renders one equation as its two sides.
    definition_rows : callable
        Renders every ``definitions`` entry of a parsed file.
    """

    escape: Callable[[str], str]
    math: Callable[[str], str]
    times: str
    column: "Callable[[_Column], str]"
    render_symbol: Callable[..., str]
    render_name: Callable[..., str]
    equation_sides: Callable[..., tuple[str, str]]
    definition_rows: Callable[..., list[tuple[str, str, str]]]


@dataclass(frozen=True)
class _Column:
    """
    One column of a tabular table: its heading, its shape, and how to read it off a row.

    Parameters
    ----------
    heading : str
        Column heading, printed in bold.
    align : str
        ``"left"`` or ``"right"``, which each dialect spells its own way.
    value : callable
        Maps a dialect and a row to the cell's markup, escaping or setting math through the dialect.
    width : float, optional
        Fraction of the text width, for a column of prose that should wrap. Defaults to sizing to content.
    """

    heading: str
    align: Literal["left", "right"]
    value: Callable[["_Dialect", Any], str]
    width: float | None = None


def typst_escape(text: str) -> str:
    """Escape the characters Typst reads as markup when they appear in content."""
    for character in "\\#$@<>*_`[]":
        text = text.replace(character, f"\\{character}")
    return text


def _latex_column(column: _Column) -> str:
    if column.width is None:
        return "r" if column.align == "right" else "l"
    return rf"p{{{column.width}\linewidth}}"


def _typst_column(column: _Column) -> str:
    return "auto" if column.width is None else f"{column.width * 100:g}%"


LATEX = _Dialect(
    escape=latex_escape,
    math=lambda markup: f"${markup}$",
    times=r"\times",
    column=_latex_column,
    render_symbol=render_latex,
    render_name=render_name_latex,
    equation_sides=equation_sides_latex,
    definition_rows=definition_rows,
)

TYPST = _Dialect(
    escape=typst_escape,
    math=lambda markup: f"${markup}$",
    times="times",
    column=_typst_column,
    render_symbol=render_typst,
    render_name=render_name_typst,
    equation_sides=equation_sides_typst,
    definition_rows=definition_rows_typst,
)

# The markup a writer emits, which is also the language a table's rows are built in.
DIALECTS: dict[Markup, _Dialect] = {"latex": LATEX, "typst": TYPST}


def _printed_columns(
    rows: Sequence[Any],
    columns: Sequence[_Column],
    dialect: _Dialect,
    widths: Sequence[str] | None,
    specifier_name: str,
) -> tuple[dict[str, list[str]], list[_Column]]:
    """
    Render every cell and return them with the columns that are not empty for every row.

    Parameters
    ----------
    rows : sequence
        The table rows, in print order.
    columns : sequence of _Column
        The columns to consider, in print order.
    dialect : _Dialect
        The language the cells are written in.
    widths : sequence of str, optional
        The caller's column specifiers, checked against the number of columns actually printed.
    specifier_name : str
        What this language calls one, for the error message.

    Returns
    -------
    cells : dict mapping str to list of str
        The rendered cells of every column, keyed by heading.
    kept : list of _Column
        The columns that have content, in print order.

    Raises
    ------
    ValueError
        If ``widths`` does not give one specifier per printed column. Checked before the caller's empty-rows
        return, so a wrong count is an error whether or not the model happens to have rows.
    """
    cells = {column.heading: [column.value(dialect, row) for row in rows] for column in columns}
    kept = [column for column in columns if any(cells[column.heading])]
    if widths is not None and len(widths) != len(kept):
        raise ValueError(
            f"widths must give one {specifier_name} per printed column, got {len(widths)} for {len(kept)}."
        )
    return cells, kept


def _tabular(
    rows: Sequence[Any],
    columns: Sequence[_Column],
    widths: Sequence[str] | None = None,
    booktabs: bool = False,
) -> str:
    r"""
    Render rows as a LaTeX ``tabular``, dropping any column that is empty for every row.

    Parameters
    ----------
    rows : sequence
        The table rows, in print order.
    columns : sequence of _Column
        The columns to consider, in print order.
    widths : sequence of str, optional
        One LaTeX column specifier per kept column, replacing the defaults. Defaults to each column's own.
    booktabs : bool, optional
        Use ``\toprule``, ``\midrule`` and ``\bottomrule`` instead of ``\hline``. Defaults to False.

    Returns
    -------
    latex : str
        The tabular, or the empty string when there are no rows.
    """
    cells, kept = _printed_columns(rows, columns, LATEX, widths, "specifier")
    if not rows:
        return ""

    top, middle, bottom = _rules(booktabs)
    header = " & ".join(rf"\textbf{{{column.heading}}}" for column in kept)
    body = " \\\\\n".join(" & ".join(cells[column.heading][index] for column in kept) for index in range(len(rows)))
    alignment = "".join(widths) if widths is not None else "".join(LATEX.column(column) for column in kept)
    return f"\\begin{{tabular}}{{{alignment}}}\n{top}\n{header} \\\\\n{middle}\n{body} \\\\\n{bottom}\n\\end{{tabular}}"


def _typst_table(
    rows: Sequence[Any],
    columns: Sequence[_Column],
    widths: Sequence[str] | None = None,
    booktabs: bool = False,
) -> str:
    """
    Render rows as a Typst ``table``, dropping any column that is empty for every row.

    Parameters
    ----------
    rows : sequence
        The table rows, in print order.
    columns : sequence of _Column
        The columns to consider, in print order.
    widths : sequence of str, optional
        One Typst column width per kept column, such as ``30%`` or ``auto``, replacing the defaults. Defaults
        to each column's own.
    booktabs : bool, optional
        Draw a rule above the header, below it and below the last row, and nothing else, which is the
        convention in this literature. Defaults to Typst's own grid.

    Returns
    -------
    typst : str
        The table, or the empty string when there are no rows.
    """
    cells, kept = _printed_columns(rows, columns, TYPST, widths, "width")
    if not rows:
        return ""

    specifiers = widths if widths is not None else [TYPST.column(column) for column in kept]
    settings = [
        f"columns: ({', '.join(specifiers)},)",
        f"align: ({', '.join(column.align for column in kept)},)",
    ]
    if booktabs:
        # Typst draws a full grid by default. Booktabs style is three horizontal rules and nothing else.
        settings.append("stroke: none")

    lines = ["#table(", *(f"  {setting}," for setting in settings)]
    if booktabs:
        lines.append("  table.hline(),")
    lines.append("  table.header(" + ", ".join(f"[*{column.heading}*]" for column in kept) + "),")
    if booktabs:
        lines.append("  table.hline(),")
    lines += [
        "  " + ", ".join(f"[{cells[column.heading][index]}]" for column in kept) + "," for index in range(len(rows))
    ]
    if booktabs:
        lines.append("  table.hline(),")
    lines.append(")")
    return "\n".join(lines)


def _wrap_typst(body: str, caption: str | None, label: str | None, size: str | None) -> str:
    """
    Wrap a rendered body in the Typst a document needs around it.

    A caption or a label puts the body in a ``figure``, because that is what Typst numbers, lists and lets
    ``@name`` refer to.

    Parameters
    ----------
    body : str
        The rendered table, starting with ``#``. An empty body wraps to nothing, because a caption on no
        table still numbers one.
    caption : str, optional
        Caption text, passed through as Typst so it can carry math. Defaults to no caption.
    label : str, optional
        Label for ``@``. Defaults to no label.
    size : str, optional
        A Typst text size, such as ``10pt``. Defaults to the document's size.

    Returns
    -------
    typst : str
        The wrapped body.
    """
    if not body:
        return ""
    if caption is None and label is None:
        return f"#text(size: {size})[\n{body}\n]" if size else body

    # A figure's arguments are read in code mode, where a call carries no leading hash.
    nested = textwrap.indent(body.removeprefix("#"), "  ").lstrip()
    arguments = [nested]
    if caption is not None:
        arguments.append(f"caption: [{caption}]")
    figure = "#figure(\n" + ",\n".join(f"  {argument}" for argument in arguments) + ",\n)"
    if label is not None:
        figure = f"{figure} <{label}>"
    return f"#text(size: {size})[\n{figure}\n]" if size else figure


@dataclass(frozen=True)
class EquationRow:
    """
    One equation of a model, rendered and ready to print.

    ``block`` is the identifier as the file spells it. The renderers turn it into prose. A ``definitions``
    entry has no ``equation_id``, because the solver substitutes it away before equations are given ids.
    ``foc_control`` is the rendered control a first-order condition was taken with respect to, and is None
    for every other kind of row.
    """

    block: str
    equation_id: str | None
    label: str | None
    left: str
    right: str
    foc_control: str | None = None


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

        A row's caption becomes a ``\tag``. An uncaptioned first-order condition is prefixed with the
        derivative it came from instead, aligned so that its ``\implies`` falls in the same column as every
        other row's ``=``. Headings use ``\intertext``, which is how amsmath interjects prose into an aligned
        block without breaking the alignment.

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
            if row.label:
                tag = rf" \tag{{\text{{{latex_escape(row.label)}}}}}"
                lines.append((False, rf"{row.left} &= {row.right}{tag}"))
            elif row.foc_control:
                derivative = rf"\frac{{\partial \mathcal{{L}}}}{{\partial {row.foc_control}}} = 0"
                lines.append((False, rf"{derivative} &\implies {row.left} = {row.right}"))
            else:
                lines.append((False, rf"{row.left} &= {row.right}"))

        # A heading is not a row and takes no terminator, and one after the final row is a LaTeX error.
        last_row = max((index for index, (is_heading, _) in enumerate(lines) if not is_heading), default=-1)
        body = "\n".join(
            text if is_heading or index == last_row else text + r" \\" for index, (is_heading, text) in enumerate(lines)
        )
        return f"\\begin{{align}}\n{body}\n\\end{{align}}"

    def to_typst(self) -> str:
        """
        Render the table as Typst, one aligned math block per model block under a heading of its own.

        Typst has no ``intertext``, so a block's heading is ordinary content between two math blocks rather
        than a line inside one. A row's caption follows its equation after a ``quad``, since Typst has no
        ``tag``.

        The rows must have been built for Typst, which :meth:`gEconpy.model.model.Model.write_table` does when
        asked for the ``typst`` writer.

        Returns
        -------
        typst : str
            The blocks, ready to paste into a document.
        """
        if not self.rows:
            return ""

        sections: list[str] = []
        rows: list[str] = []
        heading = None
        for row in self.rows:
            if row.block != heading:
                if rows:
                    sections.append(_typst_equations(rows))
                    rows = []
                heading = row.block
                sections.append(f"*{typst_escape(block_heading(heading))}*")
            rows.append(_typst_equation(row))
        sections.append(_typst_equations(rows))
        return "\n\n".join(sections)


def _typst_equation(row: EquationRow) -> str:
    """Render one row as a Typst math line, aligned on its relation."""
    if row.label:
        # A math block is sized to its content, so there is no free space for an h(1fr) to push a caption into.
        return f'{row.left} &= {row.right} quad "{row.label}"'
    if row.foc_control:
        return f"frac(partial cal(L), partial {row.foc_control}) = 0 &=> {row.left} = {row.right}"
    return f"{row.left} &= {row.right}"


def _typst_equations(rows: Sequence[str]) -> str:
    """Join rendered math lines into one Typst block, which separates its rows with a backslash."""
    return "$ " + " \\\n  ".join(rows) + " $"


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

    def to_latex(
        self,
        caption: str | None = None,
        label: str | None = None,
        size: str | None = None,
        widths: Sequence[str] | None = None,
        booktabs: bool = False,
    ) -> str:
        r"""
        Render the table as a LaTeX ``tabular``, dropping any column that is empty for every row.

        Parameters
        ----------
        caption : str, optional
            Caption text, which puts the table in a ``table`` environment. It is passed through as LaTeX,
            so it can carry math. Defaults to no caption.
        label : str, optional
            Label for ``\ref``, which puts the table in a ``table`` environment. Defaults to no label.
        size : str, optional
            A LaTeX size command without its backslash, such as ``small``. Defaults to the document's size.
        widths : sequence of str, optional
            One column specifier per printed column. Defaults to each column's own.
        booktabs : bool, optional
            Use booktabs rules instead of ``\hline``. Defaults to False.

        Returns
        -------
        latex : str
            The tabular, ready to paste into a document.
        """
        return _wrap(
            _tabular(self.rows, self._columns(), widths=widths, booktabs=booktabs),
            caption=caption,
            label=label,
            size=size,
        )

    def to_typst(
        self,
        caption: str | None = None,
        label: str | None = None,
        size: str | None = None,
        widths: Sequence[str] | None = None,
        booktabs: bool = False,
    ) -> str:
        """
        Render the table as a Typst ``table``, dropping any column that is empty for every row.

        The rows must have been built for Typst, which :meth:`gEconpy.model.model.Model.write_table` does when
        asked for the ``typst`` writer. Rendering LaTeX rows here prints LaTeX inside a Typst document.

        Parameters
        ----------
        caption : str, optional
            Caption text, which puts the table in a ``figure``. It is passed through as Typst, so it can carry
            math. Defaults to no caption.
        label : str, optional
            Label for ``@``, which puts the table in a ``figure``. Defaults to no label.
        size : str, optional
            A Typst text size, such as ``10pt``. Defaults to the document's size.
        widths : sequence of str, optional
            One Typst column width per printed column, such as ``30%``. Defaults to each column's own.
        booktabs : bool, optional
            Draw three horizontal rules and no grid. Defaults to Typst's own grid.

        Returns
        -------
        typst : str
            The table, ready to paste into a document.
        """
        return _wrap_typst(
            _typst_table(self.rows, self._columns(), widths=widths, booktabs=booktabs),
            caption=caption,
            label=label,
            size=size,
        )


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
            _Column("Symbol", "left", lambda dialect, row: dialect.math(row.symbol)),
            _Column("Description", "left", lambda dialect, row: dialect.escape(row.description or ""), width=0.5),
        ]


@dataclass(frozen=True)
class ParameterRow:
    """
    One parameter of a model, with everything a calibration table prints about it.

    ``prior`` is the distribution itself rather than a rendering of it, so a renderer can print the family,
    its moments, its parameters, or any combination.
    """

    symbol: str
    description: str | None
    value: float | None
    prior: Distribution | None
    source: str | None

    @property
    def prior_family(self) -> str:
        """Name of the prior's distribution family, empty when the parameter has no prior."""
        return "" if self.prior is None else type(self.prior).__name__

    def prior_stat(self, name: str) -> float | None:
        """
        Return one statistic of the prior.

        Parameters
        ----------
        name : str
            Any statistic the distribution computes, such as ``mean``, ``std``, ``median``, ``skewness`` or
            ``kurtosis``.

        Returns
        -------
        value : float or None
            The statistic, or None when the parameter has no prior.

        Raises
        ------
        AttributeError
            If the prior's family does not provide ``name``.
        """
        if self.prior is None:
            return None
        statistic = getattr(self.prior, name, None)
        if statistic is None:
            raise AttributeError(f"{type(self.prior).__name__} has no statistic {name!r}.")
        return float(statistic())

    @property
    def prior_parameters(self) -> str:
        """
        The prior's own parameters as ``name=value`` pairs, empty when the parameter has no prior.

        The values carry ``PRIOR_STAT_PRECISION`` figures, matching the moment columns beside them.
        """
        # An unfrozen distribution, such as a ``Beta()`` declared with no parameters, has none to report.
        params = None if self.prior is None else self.prior.params_dict
        if not params:
            return ""
        return ", ".join(f"{name}={value:.{PRIOR_STAT_PRECISION}g}" for name, value in params.items())


@dataclass(frozen=True)
class CalibrationTable(_TabularTable):
    """
    A model's parameters as table data, rendered on demand.

    Parameters
    ----------
    rows : sequence of ParameterRow
        The parameters, in print order.
    prior_stats : sequence of str, optional
        Statistics of the prior to print, one column each, from ``PRIOR_STATS``. Defaults to the mean and the
        standard deviation, which is what a DSGE paper reports.
    include_prior_params : bool, optional
        Print the distribution's own parameters in a column of their own. Defaults to False.
    """

    rows: tuple[ParameterRow, ...] | list[ParameterRow]
    prior_stats: Sequence[str] = DEFAULT_PRIOR_STATS
    include_prior_params: bool = False

    _row_type: ClassVar[type] = ParameterRow

    def __post_init__(self) -> None:
        """Reject an unknown statistic here rather than at render time, where it depends on the data."""
        unknown = [name for name in self.prior_stats if name not in PRIOR_STATS]
        if unknown:
            raise ValueError(
                f"prior_stats must name statistics from {', '.join(PRIOR_STATS)}, got {', '.join(unknown)}."
            )

    def to_frame(self) -> pd.DataFrame:
        """
        Return the table as a dataframe, one row per parameter.

        The prior is expanded into its family, moments and parameters, because a dataframe cell holding a
        live distribution object is not data anyone can work with.

        Returns
        -------
        frame : pandas.DataFrame
            Columns ``symbol``, ``description``, ``value``, ``prior``, ``prior_mean``, ``prior_sd``,
            ``prior_parameters`` and ``source``.
        """
        return pd.DataFrame(
            [
                {
                    "symbol": row.symbol,
                    "description": row.description,
                    "value": row.value,
                    "prior": row.prior_family or None,
                    **{f"prior_{name}": row.prior_stat(name) for name in self.prior_stats},
                    "prior_parameters": row.prior_parameters or None,
                    "source": row.source,
                }
                for row in self.rows
            ],
            columns=[
                "symbol",
                "description",
                "value",
                "prior",
                *(f"prior_{name}" for name in self.prior_stats),
                "prior_parameters",
                "source",
            ],
        )

    def _prior_columns(self) -> list[_Column]:
        """Return the family column, one column per requested statistic, and the parameters when asked for."""
        columns = [_Column("Prior", "left", lambda dialect, row: dialect.escape(row.prior_family))]
        columns += [
            _Column(
                PRIOR_STAT_HEADINGS.get(name, name.title()),
                "right",
                lambda dialect, row, name=name: _format_value(
                    row.prior_stat(name), precision=PRIOR_STAT_PRECISION, times=dialect.times
                ),
            )
            for name in self.prior_stats
        ]
        if self.include_prior_params:
            columns.append(
                _Column("Parameters", "left", lambda dialect, row: dialect.escape(row.prior_parameters), width=0.22)
            )
        return columns

    def _columns(self) -> list[_Column]:
        # Prose columns wrap; a symbol or a number does not need to.
        return [
            _Column("Parameter", "left", lambda dialect, row: dialect.math(row.symbol)),
            _Column("Description", "left", lambda dialect, row: dialect.escape(row.description or ""), width=0.3),
            _Column("Value", "left", lambda dialect, row: _format_value(row.value, times=dialect.times)),
            *self._prior_columns(),
            _Column("Source", "left", lambda dialect, row: dialect.escape(row.source or ""), width=0.2),
        ]
