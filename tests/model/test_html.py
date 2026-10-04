import re

import pytest

from gEconpy.data import get_example_gcn
from gEconpy.model.html import render_gcn_file
from tests.conftest import TEST_GCNS

# Sphinx tells MathJax to re-enter an ignored subtree only for these, so the container must carry one of them.
SPHINX_TYPESET_CLASSES = ("tex2jax_process", "mathjax_process", "math", "output_area")

# Variables, shocks and parameters each get a section of their own before the authored blocks.
SUMMARY_SECTIONS = 3


@pytest.fixture(scope="module")
def rendered():
    """Render once: parsing the file dominates the runtime of every assertion here."""
    return render_gcn_file(get_example_gcn("RBC"))
