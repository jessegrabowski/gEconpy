import textwrap

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from functools import partial
from typing import Any, ClassVar, Literal, Protocol, cast, get_args

import arviz_stats as azs
import numpy as np
import pandas as pd
import sympy as sp
import xarray as xr

from arviz_base import rcParams as arviz_rc_params
from preliz.distributions.distributions import Distribution
from scipy import stats
from sympy.printing.latex import latex_escape

from gEconpy.classes.time_aware_symbol import (
    render_latex,
    render_name_latex,
    render_name_typst,
    render_typst,
)
from gEconpy.model.latex import block_heading, definition_rows, equation_sides_latex, render_sides
from gEconpy.model.typst import definition_rows_typst, equation_sides_typst, typst

TableGroup = Literal[
    "equations", "variables", "shocks", "parameters", "estimates", "steady_state", "steady_state_estimates"
]
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
    render_sides : callable
        Renders a left and a right expression, taking the declared overrides.
    """

    escape: Callable[[str], str]
    math: Callable[[str], str]
    times: str
    column: "Callable[[_Column], str]"
    render_symbol: Callable[..., str]
    render_name: Callable[..., str]
    equation_sides: Callable[..., tuple[str, str]]
    definition_rows: Callable[..., list[tuple[str, str, str, str]]]
    render_sides: Callable[..., tuple[str, str]]


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
    render_sides=partial(render_sides, printer=sp.latex, renderer=render_latex),
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
    render_sides=partial(render_sides, printer=typst, renderer=render_typst),
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
    header = " & ".join(rf"\textbf{{{latex_escape(column.heading)}}}" for column in kept)
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
    lines.append("  table.header(" + ", ".join(f"[*{typst_escape(column.heading)}*]" for column in kept) + "),")
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
    entry has the id ``{block}.definitions.{position}``, which is absent from ``Model.equation_ids`` because the
    solver substitutes definitions away. ``foc_control`` is the rendered control a first-order condition was taken
    with respect to, and is None for every other kind of row.
    """

    block: str
    equation_id: str
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
    """
    One variable or shock of a model, with the caption its ``symbols`` entry declares.

    ``steady_state`` is the variable's steady-state value, None for a shock or a table built without one.
    """

    symbol: str
    description: str | None
    steady_state: float | None = None


@dataclass(frozen=True)
class SymbolTable(_TabularTable):
    """A model's variables or shocks as table data, rendered on demand."""

    rows: tuple[SymbolRow, ...] | list[SymbolRow]

    _row_type: ClassVar[type] = SymbolRow

    def _columns(self) -> list[_Column]:
        return [
            _Column("Symbol", "left", lambda dialect, row: dialect.math(row.symbol)),
            _Column("Description", "left", lambda dialect, row: dialect.escape(row.description or ""), width=0.5),
            _Column("Steady state", "right", lambda dialect, row: _format_value(row.steady_state, times=dialect.times)),
        ]


def _format_statistic(dialect: "_Dialect", value: float | None) -> str:
    return _format_value(value, precision=PRIOR_STAT_PRECISION, times=dialect.times)


def _prior_family(prior: Distribution | None) -> str:
    return "" if prior is None else type(prior).__name__


def _prior_stat(prior: Distribution | None, name: str) -> float | None:
    if prior is None:
        return None
    statistic = getattr(prior, name, None)
    if statistic is None:
        raise AttributeError(f"{type(prior).__name__} has no statistic {name!r}.")
    return float(statistic())


def _prior_parameters(prior: Distribution | None) -> str:
    # An unfrozen distribution, such as a ``Beta()`` declared with no parameters, has none to report.
    params = None if prior is None else prior.params_dict
    if not params:
        return ""
    return ", ".join(f"{name}={value:.{PRIOR_STAT_PRECISION}g}" for name, value in params.items())


class _PriorReader(Protocol):
    @property
    def prior_family(self) -> str: ...

    def prior_stat(self, name: str) -> float | None: ...

    @property
    def prior_parameters(self) -> str: ...


def _check_prior_stats(prior_stats: Sequence[str]) -> None:
    """Reject an unknown statistic when a table is built rather than at render time, where it depends on the data."""
    unknown = [name for name in prior_stats if name not in PRIOR_STATS]
    if unknown:
        raise ValueError(f"prior_stats must name statistics from {', '.join(PRIOR_STATS)}, got {', '.join(unknown)}.")


def _prefixed_heading(prefix: str, heading: str) -> str:
    """Join a prefix to a heading in sentence case, keeping an abbreviation such as ``S.D.`` as it is."""
    if not prefix:
        return heading
    return prefix + (heading if heading.isupper() else heading.lower())


def _prior_columns(prior_stats: Sequence[str], include_prior_params: bool, stat_prefix: str = "") -> list[_Column]:
    """
    Return the family column, one column per requested statistic, and the parameters when asked for.

    ``stat_prefix`` heads each statistic, for a table with no family column to say the statistics are the prior's.
    """
    columns = [_Column("Prior", "left", lambda dialect, row: dialect.escape(row.prior_family))]
    columns += [
        _Column(
            _prefixed_heading(stat_prefix, PRIOR_STAT_HEADINGS.get(name, name.title())),
            "right",
            lambda dialect, row, name=name: _format_statistic(dialect, row.prior_stat(name)),
        )
        for name in prior_stats
    ]
    if include_prior_params:
        columns.append(
            _Column("Parameters", "left", lambda dialect, row: dialect.escape(row.prior_parameters), width=0.22)
        )
    return columns


def _prior_frame_columns(prior_stats: Sequence[str]) -> list[str]:
    """Return the dataframe columns a prior expands into, in print order."""
    return ["prior", *(f"prior_{name}" for name in prior_stats), "prior_parameters"]


def _prior_frame_values(row: _PriorReader, prior_stats: Sequence[str]) -> dict[str, Any]:
    """
    Return a row's prior as dataframe cells, keyed by :func:`_prior_frame_columns`.

    The prior is expanded into its family, moments and parameters, because a dataframe cell holding a live
    distribution object is not data anyone can work with.
    """
    return {
        "prior": row.prior_family or None,
        **{f"prior_{name}": row.prior_stat(name) for name in prior_stats},
        "prior_parameters": row.prior_parameters or None,
    }


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
        return _prior_family(self.prior)

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
        return _prior_stat(self.prior, name)

    @property
    def prior_parameters(self) -> str:
        """
        The prior's own parameters as ``name=value`` pairs, empty when the parameter has no prior.

        The values carry ``PRIOR_STAT_PRECISION`` figures, matching the moment columns beside them.
        """
        return _prior_parameters(self.prior)


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
        _check_prior_stats(self.prior_stats)

    def to_frame(self) -> pd.DataFrame:
        """
        Return the table as a dataframe, one row per parameter.

        Returns
        -------
        frame : pandas.DataFrame
            Columns ``symbol``, ``description``, ``value``, ``prior``, one ``prior_<stat>`` per requested
            statistic, ``prior_parameters`` and ``source``.
        """
        return pd.DataFrame(
            [
                {
                    "symbol": row.symbol,
                    "description": row.description,
                    "value": row.value,
                    **_prior_frame_values(row, self.prior_stats),
                    "source": row.source,
                }
                for row in self.rows
            ],
            columns=["symbol", "description", "value", *_prior_frame_columns(self.prior_stats), "source"],
        )

    def _columns(self) -> list[_Column]:
        # Prose columns wrap; a symbol or a number does not need to.
        return [
            _Column("Parameter", "left", lambda dialect, row: dialect.math(row.symbol)),
            _Column("Description", "left", lambda dialect, row: dialect.escape(row.description or ""), width=0.3),
            _Column("Value", "left", lambda dialect, row: _format_value(row.value, times=dialect.times)),
            *_prior_columns(self.prior_stats, include_prior_params=self.include_prior_params),
            _Column("Source", "left", lambda dialect, row: dialect.escape(row.source or ""), width=0.2),
        ]


POSTERIOR_FRAME_COLUMNS = ("posterior_mean", "posterior_sd", "posterior_lower", "posterior_upper")
DIAGNOSTIC_FRAME_COLUMNS = ("mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat")
SAMPLE_DIMS = ("chain", "draw")


def posterior_dataset(idata: xr.DataTree | xr.Dataset) -> xr.Dataset:
    """Return the ``posterior`` group of sampler output, or the dataset itself when it is already a posterior."""
    if not isinstance(idata, xr.DataTree):
        return idata
    if "posterior" not in idata.children:
        groups = ", ".join(idata.children) or "none"
        raise ValueError(f"idata has no posterior group to summarize. Its groups are: {groups}.")
    return idata["posterior"].to_dataset()


def summarize_posterior(
    posterior: xr.Dataset,
    names: Sequence[str],
    ci_kind: Literal["hdi", "eti"] = "hdi",
    ci_prob: float | None = None,
    diagnostics: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Summarize the posterior of each named scalar variable.

    Parameters
    ----------
    posterior : Dataset
        Posterior draws, as :func:`posterior_dataset` returns them.
    names : sequence of str
        Variables to summarize. Each must be in the posterior with only ``chain`` and ``draw`` dimensions.
    ci_kind : str, optional
        ``"hdi"`` for the highest density interval or ``"eti"`` for the equal-tailed one. Defaults to ``"hdi"``.
    ci_prob : float, optional
        Probability mass of the interval. Defaults to arviz's ``stats.ci_prob``.
    diagnostics : bool, optional
        Add the convergence diagnostics, ``DIAGNOSTIC_FRAME_COLUMNS``. Defaults to False.

    Returns
    -------
    summary : pandas.DataFrame
        One row per name, with columns ``POSTERIOR_FRAME_COLUMNS`` and, on request, ``DIAGNOSTIC_FRAME_COLUMNS``.
    interval : str
        The interval's heading, such as ``"89% HDI"``.
    """
    vector = [name for name in names if posterior[name].dims != SAMPLE_DIMS]
    if vector:
        raise ValueError(
            f"A table row holds one scalar, but these have dimensions beyond chain and draw: {', '.join(vector)}."
        )

    mass = float(arviz_rc_params["stats.ci_prob"]) if ci_prob is None else ci_prob
    # The wide format, which is the default, is always a dataframe.
    summary = cast(
        pd.DataFrame,
        azs.summary(
            posterior,
            var_names=list(names),
            kind="all" if diagnostics else "stats",
            ci_kind=ci_kind,
            ci_prob=mass,
            round_to="none",
        ),
    )
    # arviz names the interval bounds after its kind and mass, so they are renamed by position.
    summary.columns = [*POSTERIOR_FRAME_COLUMNS, *summary.columns[len(POSTERIOR_FRAME_COLUMNS) :]]
    columns = [*POSTERIOR_FRAME_COLUMNS, *(DIAGNOSTIC_FRAME_COLUMNS if diagnostics else ())]
    return summary.loc[:, columns], f"{mass * 100:g}% {ci_kind.upper()}"


def prior_dataset(idata: xr.DataTree | xr.Dataset) -> xr.Dataset | None:
    """Return the ``prior`` group of sampler output, or None when it has none or is a bare posterior dataset."""
    if not isinstance(idata, xr.DataTree) or "prior" not in idata.children:
        return None
    return idata["prior"].to_dataset()


# The mode and the entropy of a sample need a density estimate, so a prior known only from draws prints neither.
_DRAW_STATISTICS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": np.mean,
    "median": np.median,
    "std": partial(np.std, ddof=1),
    "var": partial(np.var, ddof=1),
    "skewness": stats.skew,
    "kurtosis": stats.kurtosis,
}
# Four Monte Carlo standard errors, so a declared prior that was sampled as declared is almost never rejected.
_PRIOR_AGREEMENT_ERRORS = 4


def _draw_statistics(draws: np.ndarray, names: Sequence[str]) -> dict[str, float | None]:
    """Return each named statistic of a sample of prior draws, None for one a sample cannot estimate."""
    return {name: float(_DRAW_STATISTICS[name](draws)) if name in _DRAW_STATISTICS else None for name in names}


def _draws_match_prior(prior: Distribution, draws: np.ndarray) -> bool:
    """
    Report whether prior draws are consistent with a declared prior.

    The draws match when their mean and standard deviation are each within ``_PRIOR_AGREEMENT_ERRORS`` Monte
    Carlo standard errors of the declared prior's. A prior with no parameters, or with an infinite mean or
    standard deviation, never matches, because its moments cannot be compared.
    """
    if not prior.params_dict:
        return False
    declared_mean, declared_std = float(prior.mean()), float(prior.std())
    if not (np.isfinite(declared_mean) and np.isfinite(declared_std)):
        return False

    sample_std = float(np.std(draws, ddof=1))
    mean_error = sample_std / np.sqrt(draws.size)
    std_error = sample_std * np.sqrt((stats.kurtosis(draws) + 2) / (4 * draws.size))
    return (
        abs(declared_mean - float(np.mean(draws))) <= _PRIOR_AGREEMENT_ERRORS * mean_error
        and abs(declared_std - sample_std) <= _PRIOR_AGREEMENT_ERRORS * std_error
    )


def reconcile_prior(
    declared: Distribution | None, draws: np.ndarray | None, prior_stats: Sequence[str]
) -> tuple[Distribution | None, dict[str, float | None] | None]:
    """
    Choose what a row reports as its prior: the declared distribution, or the statistics of its draws.

    Draws are the prior that was actually sampled, so they win unless they match the declaration, in which case
    the declaration's exact moments and family print instead.

    Parameters
    ----------
    declared : Distribution, optional
        The prior the GCN file declares, if any.
    draws : ndarray, optional
        Draws of the prior from the sampler output's ``prior`` group, if any.
    prior_stats : sequence of str
        Statistics the table prints.

    Returns
    -------
    prior : Distribution or None
        The declared prior, when it is the one that was sampled or no draws exist.
    draw_statistics : dict of str to float, or None
        The requested statistics of the draws, when they are what the row reports.
    """
    if draws is None or (declared is not None and _draws_match_prior(declared, draws)):
        return declared, None
    return None, _draw_statistics(draws, prior_stats)


def _interval_cell(dialect: _Dialect, lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return ""
    bounds = f"{_format_statistic(dialect, lower)}, {_format_statistic(dialect, upper)}"
    return f"{dialect.escape('[')}{bounds}{dialect.escape(']')}"


def _format_fixed(value: float | None, decimals: int) -> str:
    """Render a diagnostic to fixed decimals, so an ESS of 1001 stays whole and an R-hat of 1.004 reads 1.00."""
    return "" if value is None else f"{value:.{decimals}f}"


@dataclass(frozen=True)
class EstimateRow:
    """
    One estimated parameter of a model, with its prior and, once sampled, its posterior summary.

    The prior is either a declared distribution, ``prior``, or the statistics of its draws,
    ``prior_draw_statistics``, for a prior that was declared nowhere or was sampled other than as declared.
    """

    symbol: str
    description: str | None
    prior: Distribution | None
    prior_draw_statistics: Mapping[str, float | None] | None = None
    posterior_mean: float | None = None
    posterior_sd: float | None = None
    posterior_lower: float | None = None
    posterior_upper: float | None = None
    mcse_mean: float | None = None
    mcse_sd: float | None = None
    ess_bulk: float | None = None
    ess_tail: float | None = None
    r_hat: float | None = None

    @property
    def prior_family(self) -> str:
        """Name of the declared prior's family, empty when the prior is known only from draws or not at all."""
        return _prior_family(self.prior)

    def prior_stat(self, name: str) -> float | None:
        """Return one statistic of the prior, from its draws when it has them and its declaration otherwise."""
        if self.prior_draw_statistics is not None:
            return self.prior_draw_statistics[name]
        return _prior_stat(self.prior, name)

    @property
    def prior_parameters(self) -> str:
        """The declared prior's own parameters as ``name=value`` pairs, empty without a declared prior."""
        return _prior_parameters(self.prior)


@dataclass(frozen=True)
class EstimationTable(_TabularTable):
    """
    A model's estimated parameters as table data, rendered on demand.

    Parameters
    ----------
    rows : sequence of EstimateRow
        The estimated parameters, in print order.
    prior_stats : sequence of str, optional
        Statistics of the prior to print, one column each, from ``PRIOR_STATS``. Defaults to the mean and the
        standard deviation.
    include_prior_params : bool, optional
        Print the distribution's own parameters in a column of their own. Defaults to False.
    interval : str, optional
        Heading of the posterior interval column, such as ``"89% HDI"``. Defaults to no posterior.
    symbol_heading : str, optional
        Heading of the symbol column. Defaults to ``"Parameter"``.
    prior_stat_prefix : str, optional
        Prefix of each prior statistic's heading. A table whose priors are known only from draws prints no family
        column, and passes ``"Prior "`` so its moments are not read as the posterior's. Defaults to none.
    """

    rows: tuple[EstimateRow, ...] | list[EstimateRow]
    prior_stats: Sequence[str] = DEFAULT_PRIOR_STATS
    include_prior_params: bool = False
    interval: str | None = None
    symbol_heading: str = "Parameter"
    prior_stat_prefix: str = ""

    _row_type: ClassVar[type] = EstimateRow

    def __post_init__(self) -> None:
        _check_prior_stats(self.prior_stats)

    def to_frame(self) -> pd.DataFrame:
        """
        Return the table as a dataframe, one row per estimated parameter.

        Returns
        -------
        frame : pandas.DataFrame
            Columns ``symbol``, ``description``, ``prior``, one ``prior_<stat>`` per requested statistic,
            ``prior_parameters``, ``posterior_mean``, ``posterior_sd``, ``posterior_lower``,
            ``posterior_upper``, and the convergence diagnostics ``mcse_mean``, ``mcse_sd``, ``ess_bulk``,
            ``ess_tail`` and ``r_hat``.
        """
        return pd.DataFrame(
            [
                {
                    "symbol": row.symbol,
                    "description": row.description,
                    **_prior_frame_values(row, self.prior_stats),
                    **{name: getattr(row, name) for name in (*POSTERIOR_FRAME_COLUMNS, *DIAGNOSTIC_FRAME_COLUMNS)},
                }
                for row in self.rows
            ],
            columns=[
                "symbol",
                "description",
                *_prior_frame_columns(self.prior_stats),
                *POSTERIOR_FRAME_COLUMNS,
                *DIAGNOSTIC_FRAME_COLUMNS,
            ],
        )

    def _columns(self) -> list[_Column]:
        return [
            _Column(self.symbol_heading, "left", lambda dialect, row: dialect.math(row.symbol)),
            _Column("Description", "left", lambda dialect, row: dialect.escape(row.description or ""), width=0.3),
            *_prior_columns(
                self.prior_stats, include_prior_params=self.include_prior_params, stat_prefix=self.prior_stat_prefix
            ),
            _Column("Post. mean", "right", lambda dialect, row: _format_statistic(dialect, row.posterior_mean)),
            _Column("Post. S.D.", "right", lambda dialect, row: _format_statistic(dialect, row.posterior_sd)),
            _Column(
                self.interval or "Interval",
                "right",
                lambda dialect, row: _interval_cell(dialect, row.posterior_lower, row.posterior_upper),
            ),
            _Column("MCSE mean", "right", lambda dialect, row: _format_statistic(dialect, row.mcse_mean)),
            _Column("MCSE S.D.", "right", lambda dialect, row: _format_statistic(dialect, row.mcse_sd)),
            _Column("ESS bulk", "right", lambda _, row: _format_fixed(row.ess_bulk, decimals=0)),
            _Column("ESS tail", "right", lambda _, row: _format_fixed(row.ess_tail, decimals=0)),
            _Column("R-hat", "right", lambda _, row: _format_fixed(row.r_hat, decimals=2)),
        ]
