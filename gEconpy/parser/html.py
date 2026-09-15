from pathlib import Path

from IPython.core.display_functions import display
from IPython.display import HTML

from gEconpy.model.block import Block
from gEconpy.parser.loader import load_gcn_file


def get_css() -> str:
    """
    Build the stylesheet for the HTML representation of a model.

    The layout follows the xarray HTML representation: each block is a collapsible container with an unbroken
    background.

    Returns
    -------
    css : str
        A ``<style>`` element scoped to the model container.
    """
    return r"""
    <style>
        /* Scope all styles under the #model-container */
        #model-container {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 12px;
            color: #333;
            margin: 0;
            padding: 0;
        }

        #model-container.math {
            width: 100%;
            display: block;
        }

        #model-container .model-blocks {
            padding: 0;
        }
        #model-container .model-blocks > details.block-info {
            border: none;
            padding: 0;
            margin: 0;
        }
        #model-container .model-blocks > details.block-info:not(:last-child) {
            border-bottom: 1px solid #ddd;
        }
        #model-container .model-blocks > details {
            background-color: #f9f9f9;
        }
        #model-container details.block-info > summary.block-title {
            font-weight: bold;
            cursor: pointer;
            padding: 10px;
            background-color: inherit;
            list-style: none;
            margin: 0;
        }
        #model-container details.block-info > summary.block-title:hover {
            background-color: #e9e9e9;
        }
        #model-container details.block-info > summary.block-title::before {
            content: "►";
            display: inline-block;
            margin-right: 0.5em;
            transition: transform 0.2s ease;
        }
        #model-container details.block-info[open] > summary.block-title::before {
            content: "▼";
        }
        #model-container .block-content {
            margin: 0;
            padding: 0;
        }
        #model-container details.property-details {
            margin: 0;
            padding: 0 0 0 1em;
            border: none;
        }
        #model-container details.property-details > summary {
            font-weight: bold;
            cursor: pointer;
            padding: 8px;
            background-color: #f9f9f9;
            border-bottom: 1px solid #ddd;
            list-style: none;
        }
        #model-container details.property-details > summary:hover {
            background-color: #e9e9e9;
        }
        #model-container details.property-details > summary::before {
            content: "►";
            display: inline-block;
            margin-right: 0.5em;
            transition: transform 0.2s ease;
        }
        #model-container details.property-details[open] > summary::before {
            content: "▼";
        }
        #model-container .block-content p {
            margin: 0;
            padding: 5px 10px;
        }
    </style>
    """


def generate_html(blocks: list[Block]) -> HTML:
    """
    Render model blocks as collapsible HTML with MathJax-typeset equations.

    Parameters
    ----------
    blocks : list of Block
        Blocks to render, in display order.

    Returns
    -------
    html : HTML
        An IPython display object holding the rendered model.
    """
    html_parts = []
    html_parts.append("""
    <script>
      if (typeof window.MathJax === 'undefined') {
        var mathjaxScript = document.createElement('script');
        mathjaxScript.type = 'text/javascript';
        mathjaxScript.async = true;
        mathjaxScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML';
        document.head.appendChild(mathjaxScript);
      }
    </script>
    """)
    html_parts.append(get_css())

    html_parts.append("<div id='model-container' class='math model-container-subclass'>")
    html_parts.append("<div class='model-blocks'>")
    html_parts.extend([block.__html_repr__() for block in blocks])
    html_parts.append("</div>")
    html_parts.append("</div>")

    html_parts.append("""
    <script>
      function reprocessMath() {
        if (window.MathJax && window.MathJax.Hub) {
          MathJax.Hub.Queue(["Typeset", MathJax.Hub, document.getElementById("model-container")]);
        } else {
          setTimeout(reprocessMath, 100);
        }
      }
      reprocessMath();
    </script>
    """)

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
