from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, Literal, get_args

import pandas as pd

from preliz.distributions.distributions import Distribution
from sympy.printing.latex import latex_escape

from gEconpy.model.latex import block_heading

TableGroup = Literal["equations", "variables", "shocks", "parameters"]
# The statistics every preliz family reports, which is the vocabulary a caller may ask a prior for. A name
# outside it is rejected rather than rendered empty, because an empty column in a paper is not a visible error.
PRIOR_STATS = ("mean", "median", "mode", "std", "var", "skewness", "kurtosis", "entropy")
# Headings for the names a title case would get wrong.
PRIOR_STAT_HEADINGS = {"std": "S.D.", "var": "Variance", "skewness": "Skew", "kurtosis": "Kurtosis"}
DEFAULT_PRIOR_STATS = ("mean", "std")
# A prior mean of 0.343849 is noise in a paper, and three figures is what the literature prints.
PRIOR_STAT_PRECISION = 3
TABLE_GROUPS: tuple[TableGroup, ...] = get_args(TableGroup)
# Writer name to the method that renders it. ``test_every_announced_writer_renders_every_group`` parametrizes
# over this mapping and every group, so a writer added here without a method on all three tables fails.
TABLE_WRITERS = {"latex": "to_latex"}


def _format_value(value: float | None, math: tuple[str, str] = ("$", "$"), precision: int = 6) -> str:
    """
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
    return rf"{opening}{mantissa} \times 10^{{{int(exponent)}}}{closing}"


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
    cells = {column.heading: [column.value(row) for row in rows] for column in columns}
    kept = [column for column in columns if any(cells[column.heading])]
    # Checked before the empty-rows return, so a wrong column count is an error whether or not the model
    # happens to have rows.
    if widths is not None and len(widths) != len(kept):
        raise ValueError(f"widths must give one specifier per printed column, got {len(widths)} for {len(kept)}.")

    if not rows:
        return ""

    top, middle, bottom = _rules(booktabs)
    header = " & ".join(rf"\textbf{{{column.heading}}}" for column in kept)
    body = " \\\\\n".join(" & ".join(cells[column.heading][index] for column in kept) for index in range(len(rows)))
    alignment = "".join(widths) if widths is not None else "".join(column.alignment for column in kept)
    return f"\\begin{{tabular}}{{{alignment}}}\n{top}\n{header} \\\\\n{middle}\n{body} \\\\\n{bottom}\n\\end{{tabular}}"


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
        columns = [_Column("Prior", "l", lambda row: latex_escape(row.prior_family))]
        columns += [
            _Column(
                PRIOR_STAT_HEADINGS.get(name, name.title()),
                "r",
                lambda row, name=name: _format_value(row.prior_stat(name), precision=PRIOR_STAT_PRECISION),
            )
            for name in self.prior_stats
        ]
        if self.include_prior_params:
            columns.append(_Column("Parameters", r"p{0.22\linewidth}", lambda row: latex_escape(row.prior_parameters)))
        return columns

    def _columns(self) -> list[_Column]:
        # Prose columns wrap; a symbol or a number does not need to.
        return [
            _Column("Parameter", "l", lambda row: f"${row.symbol}$"),
            _Column("Description", r"p{0.3\linewidth}", lambda row: latex_escape(row.description or "")),
            _Column("Value", "l", lambda row: _format_value(row.value)),
            *self._prior_columns(),
            _Column("Source", r"p{0.2\linewidth}", lambda row: latex_escape(row.source or "")),
        ]
