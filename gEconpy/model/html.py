from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import sympy as sp

from IPython.core.display_functions import display
from IPython.display import HTML

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol, render_latex
from gEconpy.model.latex import authored_equation_latex, block_heading
from gEconpy.model.tables import PRIOR_STAT_PRECISION, SymbolTable, _format_value
from gEconpy.parser.ast import GCNModel, SymbolDeclaration, variable_key
from gEconpy.parser.ast.nodes import GCNBlock, GCNDistribution, GCNEquation
from gEconpy.parser.loader import load_gcn_file

if TYPE_CHECKING:
    from gEconpy.model.model import Model

# Captions the fallback steady-state section, which only a model with no parsed file renders. It differs from
# an authored block's heading, which ``block_heading`` builds from the name the author wrote.
SOLVED_STEADY_STATE_TITLE = "Steady state"


def get_css() -> str:
    """
    Build the stylesheet for the HTML representation of a model.

    The layout follows the xarray HTML representation: each block is a collapsible container with an unbroken
    background. Every color is a ``--ge-*`` variable derived from the JupyterLab ``--jp-*`` variable of the
    same role, with a pydata-sphinx-theme ``--pst-*`` variable and then a literal as fallbacks, so the
    representation tracks the active theme instead of assuming a light one. The variables are defined on the
    container rather than on ``:root``, so a rendered model writes nothing at document scope.

    Returns
    -------
    css : str
        A ``<style>`` element scoped to the ``ge-model`` class.
    """
    return r"""
    <style>
        .ge-model {
            --ge-font-color: var(--jp-content-font-color0, var(--pst-color-text-base, rgba(0, 0, 0, 0.8)));
            --ge-border-color: var(--jp-border-color2, var(--pst-color-border, rgba(0, 0, 0, 0.13)));
            --ge-background-color: var(--jp-layout-color0, var(--pst-color-on-background, white));
            --ge-background-color-section: var(--jp-layout-color1, var(--pst-color-surface, #f9f9f9));
            --ge-background-color-hover: var(--jp-layout-color2, var(--pst-color-panel-background, #e9e9e9));
        }

        html[theme="dark"] .ge-model,
        html[data-theme="dark"] .ge-model,
        body[data-theme="dark"] .ge-model,
        body[data-jp-theme-light="false"] .ge-model,
        body.vscode-dark .ge-model {
            --ge-font-color: var(--jp-content-font-color0, var(--pst-color-text-base, rgba(255, 255, 255, 0.8)));
            --ge-border-color: var(--jp-border-color2, var(--pst-color-border, rgba(255, 255, 255, 0.13)));
            --ge-background-color: var(--jp-layout-color0, var(--pst-color-on-background, #111111));
            --ge-background-color-section: var(--jp-layout-color1, var(--pst-color-surface, #1a1a1a));
            --ge-background-color-hover: var(--jp-layout-color2, var(--pst-color-panel-background, #262626));
        }

        /* Layout. Every color below is one of the variables above, never a literal. */
        .ge-model {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 12px;
            color: var(--ge-font-color);
            background-color: var(--ge-background-color);
            margin: 0;
            padding: 0;
        }

        .ge-model.math {
            width: 100%;
            display: block;
        }

        .ge-model .model-blocks {
            padding: 0;
        }
        .ge-model .model-blocks > details.block-info {
            border: none;
            padding: 0;
            margin: 0;
        }
        .ge-model .model-blocks > details.block-info:not(:last-child) {
            border-bottom: 1px solid var(--ge-border-color);
        }
        .ge-model .model-blocks > details {
            background-color: var(--ge-background-color-section);
        }
        .ge-model details.block-info > summary.block-title {
            font-weight: bold;
            cursor: pointer;
            padding: 10px;
            background-color: inherit;
            list-style: none;
            margin: 0;
        }
        .ge-model details.block-info > summary.block-title:hover {
            background-color: var(--ge-background-color-hover);
        }
        .ge-model details.block-info > summary.block-title::before {
            content: "\25BA";
            display: inline-block;
            margin-right: 0.5em;
            transition: transform 0.2s ease;
        }
        .ge-model details.block-info[open] > summary.block-title::before {
            content: "\25BC";
        }
        .ge-model .block-content {
            margin: 0;
            padding: 0;
        }
        .ge-model .block-content p {
            margin: 0;
            padding: 5px 10px;
        }
        .ge-model p.ge-subject-to,
        .ge-model p.ge-component {
            font-style: italic;
            padding: 5px 10px 0 10px;
        }
        .ge-model p.ge-declaration {
            font-family: monospace;
        }
        .ge-model table.ge-table {
            border-collapse: collapse;
            margin: 5px 10px;
        }
        .ge-model table.ge-table th {
            text-align: left;
            font-weight: bold;
            padding: 4px 12px 4px 0;
            border-bottom: 1px solid var(--ge-border-color);
        }
        .ge-model table.ge-table td {
            text-align: left;
            padding: 4px 12px 4px 0;
            vertical-align: top;
        }
    </style>
    """


def print_gcn_file(gcn_path: str | Path) -> None:
    """
    Display the blocks of a GCN file as collapsible HTML in a notebook.

    Each block shows the program its author wrote, which is what a built model's own representation shows.
    Nothing is solved, so a first-order condition does not appear.

    Parameters
    ----------
    gcn_path : str or Path
        Path to the GCN file.

    Examples
    --------
    Inspect the packaged RBC model before building it:

    .. code-block:: python

        import gEconpy as ge
        from gEconpy.data import get_example_gcn

        ge.print_gcn_file(get_example_gcn("RBC"))
    """
    display(HTML(render_gcn_file(gcn_path)))


def render_gcn_file(gcn_path: str | Path) -> str:
    """
    Render the blocks of a GCN file as HTML, without building the model.

    Parameters
    ----------
    gcn_path : str or Path
        Path to the GCN file.

    Returns
    -------
    html : str
        The rendered blocks, including the stylesheet.
    """
    source_ast = load_gcn_file(gcn_path, simplify_blocks=False).source_ast
    overrides = _declared_latex(source_ast.symbols)
    return _document("".join(_block_section(block, overrides) for block in source_ast.blocks))


def _document(body: str) -> str:
    # The ``math`` class is load-bearing in Sphinx, not decoration. A myst-nb page carries ``tex2jax_ignore``
    # on its top-level section, and MathJax re-enters only a subtree whose class Sphinx lists in
    # ``processHtmlClass``, where ``math`` is one of four.
    return f"{get_css()}\n<div class='ge-model math'>\n<div class='model-blocks'>{body}</div>\n</div>"


def _section(title: str, body: str) -> str:
    return (
        f"<details class='block-info'><summary class='block-title'>{escape(title)}</summary>"
        f"<div class='block-content'>{body}</div></details>"
    )


def _math(latex: str) -> str:
    return f"<p>\\[{latex}\\]</p>"


def _table(rows: Sequence[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    """Render a table whose cells are already HTML, because every one of them holds either math or escaped prose."""
    head = "".join(f"<th>{escape(name)}</th>" for name in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table class='ge-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _symbol_latex(symbol: TimeAwareSymbol, overrides: dict[str, str]) -> str:
    return render_latex(symbol, stem_override=overrides.get(variable_key(symbol.base_name)))


def _declared_latex(declarations: dict[str, SymbolDeclaration]) -> dict[str, str]:
    return {name: declaration.latex for name, declaration in declarations.items() if declaration.latex}


def _symbol_cells(table: SymbolTable) -> list[tuple[str, str]]:
    r"""Wrap the public symbol rows for HTML, where math takes ``\(...\)`` and prose must be escaped."""
    return [(f"\\({row.symbol}\\)", escape(row.description or "")) for row in table.rows]


def _problem_latex(block: GCNBlock, overrides: dict[str, str]) -> list[str]:
    r"""
    Render a block's optimization problem as a ``\max`` or ``\min`` over its controls.

    The author wrote an objective and a list of controls, which is a program. Printing the two under separate
    headings makes the reader reassemble it.
    """
    objective = block.objective[0]
    operator = r"\min" if objective.is_minimize else r"\max"
    controls = ", ".join(
        _symbol_latex(TimeAwareSymbol(control.name, control.time_index.value), overrides) for control in block.controls
    )
    left, right = authored_equation_latex(objective, overrides)

    lines = [_math(rf"{operator}_{{{controls}}} \quad {left} = {right}")]
    if block.constraints:
        lines.append("<p class='ge-subject-to'>subject to</p>")
        lines.extend(_math(_constraint_latex(eq, overrides)) for eq in block.constraints)
    return lines


def _constraint_latex(equation: GCNEquation, overrides: dict[str, str]) -> str:
    """Render a constraint with its named multiplier, which is part of the program the author wrote."""
    body = " = ".join(authored_equation_latex(equation, overrides))
    if equation.lagrange_multiplier is None:
        return body

    multiplier = _symbol_latex(TimeAwareSymbol(equation.lagrange_multiplier, 0), overrides)
    return rf"{body} \quad ({multiplier})"


def _labeled(label: str, entries: list[str]) -> list[str]:
    return [f"<p class='ge-component'>{label}</p>", *entries] if entries else []


def _equation_html(equation: GCNEquation, overrides: dict[str, str]) -> str:
    return _math(" = ".join(authored_equation_latex(equation, overrides)))


def _calibration_html(entries: list[GCNEquation | GCNDistribution], overrides: dict[str, str]) -> list[str]:
    """Render a calibration entry as the author wrote it, which for a prior is a declaration rather than math."""
    return [
        f"<p class='ge-declaration'>{escape(str(entry))}</p>"
        if isinstance(entry, GCNDistribution)
        else _equation_html(entry, overrides)
        for entry in entries
    ]


def _block_section(block: GCNBlock, overrides: dict[str, str]) -> str:
    """
    Render one authored block, with its components in the order a ``.gcn`` file writes them.

    A definition is written before the objective that uses it, so it prints before the program rather than
    after it.
    """
    if block.has_optimization_problem():
        parts = [
            *_labeled("Definitions", [_equation_html(eq, overrides) for eq in block.definitions]),
            *_problem_latex(block, overrides),
            *_labeled("Identities", [_equation_html(eq, overrides) for eq in block.identities]),
        ]
    else:
        parts = [_equation_html(eq, overrides) for eq in block.all_equations()]

    shocks = [
        _math(_symbol_latex(TimeAwareSymbol(shock.name, shock.time_index.value), overrides)) for shock in block.shocks
    ]
    parts.extend(_labeled("Shocks", shocks))
    parts.extend(_labeled("Calibration", _calibration_html(block.calibration, overrides)))

    return _section(block_heading(block.name), "".join(parts))


def _calibration_section(model: "Model") -> str:
    """Reuse the calibration table's rows, so the notebook view and a paper's table cannot disagree."""
    rows = model.table("parameters").rows
    cells = [
        (
            f"\\({row.symbol}\\)",
            escape(row.description or ""),
            _format_value(row.value, math=(r"\(", r"\)")),
            escape(row.prior_family),
            _format_value(row.prior_stat("mean"), math=(r"\(", r"\)"), precision=PRIOR_STAT_PRECISION),
            _format_value(row.prior_stat("std"), math=(r"\(", r"\)"), precision=PRIOR_STAT_PRECISION),
            escape(row.source or ""),
        )
        for row in rows
    ]
    # Counted from the rows, because the parameters group adds calibrated parameters and shock
    # hyper-parameters that ``model.params`` does not hold.
    return _section(
        f"Parameters ({len(rows)})",
        _table(cells, ("Symbol", "Description", "Value", "Prior", "Mean", "S.D.", "Source")),
    )


def render_model(model: "Model", source_ast: GCNModel | None = None) -> str:
    r"""
    Render a built model as HTML: its flattened system, then the blocks its author wrote.

    The two halves come from two sources and neither is derived twice. Variables, parameters, shocks and
    their declared captions come from the model. The block structure comes from the parsed file, because the
    solved blocks hold post-simplification equations and would show the author something they did not write.

    Parameters
    ----------
    model : Model
        The built model.
    source_ast : GCNModel, optional
        The parsed file, which supplies the block sections. A model built without one renders the symbol
        tables alone, since they come from the model itself.

    Returns
    -------
    html : str
        The rendered representation, including its stylesheet.
    """
    # The symbol tables take their captions from the model, not from ``source_ast``, which may be absent.
    # The block sections below need the overrides directly, because they render authored equations.
    overrides = model._latex_overrides()
    sections = [
        _section(
            f"Variables ({len(model.variables)})",
            _table(_symbol_cells(model.table("variables")), ("Symbol", "Description")),
        ),
        _section(
            f"Shocks ({len(model.shocks)})",
            _table(_symbol_cells(model.table("shocks")), ("Symbol", "Description")),
        ),
        _calibration_section(model),
    ]

    if source_ast is not None:
        # The authored steady-state block renders here too, in the form its author wrote rather than the
        # substituted form ``steady_state_relationships`` holds.
        sections.extend(_block_section(block, overrides) for block in source_ast.blocks)
    elif model.steady_state_relationships:
        sections.append(
            _section(
                SOLVED_STEADY_STATE_TITLE,
                "".join(_math(f"{sp.latex(eq.lhs)} = {sp.latex(eq.rhs)}") for eq in model.steady_state_relationships),
            )
        )

    return _document("".join(sections))
