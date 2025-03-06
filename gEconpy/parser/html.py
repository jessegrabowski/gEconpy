from typing import TYPE_CHECKING

from IPython.core.display_functions import display
from IPython.display import HTML

from gEconpy.parser.file_loaders import gcn_to_block_dict

if TYPE_CHECKING:
    from gEconpy.model.block import Block


def get_css() -> str:
    """
    Return a CSS string to style the generated HTML.
    The style is inspired by the xarray HTML representation.
    Each block is rendered in a unified container with an unbroken background,
    and the whole block is collapsible.
    """
    css = r"""
    <style>
        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 12px;
            color: #333;
            margin: 0;
            padding: 0;
        }
        /* Container for all model blocks */
        .model-blocks {
            padding: 0;
        }
        /* Top-level blocks inside the model-blocks container */
        .model-blocks > details.block-info {
            border: none;
            padding: 0;
            margin: 0;
        }
        .model-blocks > details.block-info:not(:last-child) {
            border-bottom: 1px solid #ddd;
        }
        .model-blocks > details {
            background-color: #f9f9f9;
        }
        /* Block title styling: make it span the full width and inherit background */
        details.block-info > summary.block-title {
            font-weight: bold;
            cursor: pointer;
            padding: 10px;
            background-color: inherit;
            list-style: none;
            margin: 0;
        }
        details.block-info > summary.block-title:hover {
            background-color: #e9e9e9;
        }
        details.block-info > summary.block-title::before {
            content: "►";
            display: inline-block;
            margin-right: 0.5em;
            transition: transform 0.2s ease;
        }
        details.block-info[open] > summary.block-title::before {
            content: "▼";
        }
        /* Container for the block content */
        .block-content {
            margin: 0;
            padding: 0;
        }
        /* Collapsible sub-sections within the block */
        details.property-details {
            margin: 0;
            padding: 0 0 0 1em;
            border: none;
        }
        details.property-details > summary {
            font-weight: bold;
            cursor: pointer;
            padding: 8px;
            background-color: #f9f9f9;
            border-bottom: 1px solid #ddd;
            list-style: none;
        }
        details.property-details > summary:hover {
            background-color: #e9e9e9;
        }
        details.property-details > summary::before {
            content: "►";
            display: inline-block;
            margin-right: 0.5em;
            transition: transform 0.2s ease;
        }
        details.property-details[open] > summary::before {
            content: "▼";
        }
        .block-content p {
            margin: 0;
            padding: 5px 10px;
        }
    </style>
    """
    return css


def generate_html(blocks: list["Block"]) -> HTML:
    """
    Represent a model in HTML

    Parameters
    ----------
    blocks : list[Block]
        List of blocks to represent
    """
    html_parts = []
    html_parts.append("<html><head>")
    html_parts.append("<meta charset='utf-8'>")
    html_parts.append("""
    <script type="text/javascript" async
      src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML">
    </script>
    """)
    html_parts.append(get_css())
    html_parts.append("</head><body>")

    html_parts.append("<div class='model-blocks'>")
    for block in blocks:
        html_parts.append(block.__html_repr__())
    html_parts.append("</div>")

    html_parts.append("</body></html>")

    final_html = "\n".join(html_parts)
    return HTML(final_html)


def print_gcn_file(gcn_path: str) -> None:
    """
    Display a model in HTML

    Parameters
    ----------
    gcn_file : str
        Path to the GCN file
    """
    outputs = gcn_to_block_dict(gcn_path, simplify_blocks=False, include_ss_block=True)
    block_dict, assumptions, options, try_reduce, ss_solution_dict, prior_info = outputs
    blocks = list(block_dict.values())

    # TODO: Do stuff with the other outputs
    html = generate_html(blocks)
    display(html)
