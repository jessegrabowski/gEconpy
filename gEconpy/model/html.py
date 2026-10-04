from pathlib import Path

from IPython.core.display_functions import display
from IPython.display import HTML

from gEconpy.model.block import Block
from gEconpy.parser.loader import load_gcn_file


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
            content: "\\25BA";
            display: inline-block;
            margin-right: 0.5em;
            transition: transform 0.2s ease;
        }
        .ge-model details.block-info[open] > summary.block-title::before {
            content: "\\25BC";
        }
        .ge-model .block-content {
            margin: 0;
            padding: 0;
        }
        .ge-model details.property-details {
            margin: 0;
            padding: 0 0 0 1em;
            border: none;
        }
        .ge-model details.property-details > summary {
            font-weight: bold;
            cursor: pointer;
            padding: 8px;
            background-color: var(--ge-background-color-section);
            border-bottom: 1px solid var(--ge-border-color);
            list-style: none;
        }
        .ge-model details.property-details > summary:hover {
            background-color: var(--ge-background-color-hover);
        }
        .ge-model details.property-details > summary::before {
            content: "\\25BA";
            display: inline-block;
            margin-right: 0.5em;
            transition: transform 0.2s ease;
        }
        .ge-model details.property-details[open] > summary::before {
            content: "\\25BC";
        }
        .ge-model .block-content p {
            margin: 0;
            padding: 5px 10px;
        }
    </style>
    """


def generate_html(blocks: list[Block]) -> HTML:
    r"""
    Render model blocks as collapsible HTML with equations in ``\[...\]`` delimiters.

    Nothing typesets the equations here, because every frontend that displays ``text/html`` output already
    does. JupyterLab, the classic notebook and ``nbconvert`` each run their own typesetter over HTML output,
    and a Sphinx page typesets any subtree whose class it lists in ``processHtmlClass``.

    Parameters
    ----------
    blocks : list of Block
        Blocks to render, in display order.

    Returns
    -------
    html : HTML
        An IPython display object holding the rendered model.
    """
    html_parts = [get_css()]

    # The ``math`` class is load-bearing in Sphinx, not decoration. A myst-nb page carries ``tex2jax_ignore``
    # on its top-level section, and MathJax re-enters only a subtree whose class Sphinx lists in
    # ``processHtmlClass``, where ``math`` is one of four.
    html_parts.append("<div class='ge-model math'>")
    html_parts.append("<div class='model-blocks'>")
    html_parts.extend([block.__html_repr__() for block in blocks])
    html_parts.append("</div>")
    html_parts.append("</div>")

    return HTML("\n".join(html_parts))


def print_gcn_file(gcn_path: str | Path) -> None:
    """
    Display the blocks of a GCN file as collapsible HTML in a notebook.

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
    primitives = load_gcn_file(gcn_path, simplify_blocks=False)
    display(generate_html(list(primitives.block_dict.values())))
