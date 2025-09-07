import os
import sys

from pathlib import Path

root_dir = Path("../..").resolve()
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "sphinxext"))

import gEconpy  # noqa: E402

# -- Project information -----------------------------------------------------
project = "gEconpy"
copyright = "2022-2025, Jesse Grabowski"
language = "en"
html_baseurl = "github.com/jessegrabowski/gEconpy"

docnames = []

version = gEconpy.__version__
on_readthedocs = os.environ.get("READTHEDOCS", None)
rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
if on_readthedocs:
    if rtd_version.lower() == "stable":
        version = gEconpy.__version__.split("+")[0]
    elif rtd_version.lower() == "latest":
        version = "dev"
    else:
        version = rtd_version
else:
    rtd_version = "local"
# The full version, including alpha/beta/rc tags.
release = version

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "pydata_sphinx_theme",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinx_codeautolink",
    "generate_gallery",
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
]


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "*/autosummary/*.rst",
    "Thumbs.db",
    ".DS_Store",
]

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "pydata_sphinx_theme"
html_title = project
html_short_title = project
html_last_updated_fmt = ""

rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
sitemap_url_scheme = f"{{lang}}{rtd_version}/{{link}}"
html_theme_options = {
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "navbar_start": ["navbar-logo"],
    # "article_header_end": ["nb-badges"],
    "show_prev_next": True,
    # "article_footer_items": ["rendered_citation.html"],
}
version = version if "." in rtd_version else "main"
# doi_code = os.environ.get("DOI_READTHEDOCS", "10.5281/zenodo.5654871")
html_context = {
    "github_url": "https://github.com",
    "github_user": "jessegrabowski",
    "github_repo": "gEconpy",
    "github_version": version,
    "doc_path": "docs/",
    # "sandbox_repo": f"pymc-devs/pymc-sandbox/{version}",
    # "doi_url": f"https://doi.org/{doi_code}",
    # "doi_code": doi_code,
    "default_mode": "dark",
}


# html_favicon = "../_static/PyMC.ico"
# html_logo = "../_static/PyMC.png"
html_title = "gEconpy: DSGE Modeling in Python"
html_sidebars = {"**": ["sidebar-nav-bs.html", "searchbox.html"]}

# ----Miscellaneous Config------------------------------
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

# The master toctree document.
master_doc = "index"

# Don't auto-generate summary for class members.
autosummary_generate = True
autodoc_typehints = "none"
autoclass_content = "class"
remove_from_toctrees = ["**/classmethods/*"]

numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {
    "of",
    "or",
    "optional",
    "default",
    "numeric",
    "type",
    "scalar",
    "1D",
    "2D",
    "3D",
    "nD",
    "array",
    "instance",
    "M",
    "N",
}

# -- MyST config  -------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "substitution",
]
myst_dmath_double_inline = True

myst_substitutions = {
    "pip_dependencies": "{{ extra_dependencies }}",
    "conda_dependencies": "{{ extra_dependencies }}",
    "extra_install_notes": "",
}

nb_execution_mode = "off"
nbsphinx_execute = "never"
nbsphinx_allow_errors = True


# -- Bibtex config  -------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# -- ABlog config  -------------------------------------------------
# blog_baseurl = "https://docs.pymc.io/projects/examples/en/latest/"
# blog_title = "PyMC Examples"
# blog_path = "blog"
# blog_authors = {
#     "contributors": ("PyMC Contributors", "https://docs.pymc.io"),
# }
# blog_default_author = "contributors"
# post_show_prev_next = False
# fontawesome_included = True


# -- Intersphinx Mapping -------------------------------------------------
intersphinx_mapping = {
    "arviz": ("https://python.arviz.org/en/latest/", None),
    "pytensor": ("https://pytensor.readthedocs.io/en/latest/", None),
    "pmx": ("https://www.pymc.io/projects/experimental/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "myst": ("https://myst-parser.readthedocs.io/en/latest", None),
    "myst-nb": ("https://myst-nb.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

# OpenGraph config
# use default readthedocs integration aka no config here

# codeautolink_autodoc_inject = False
# codeautolink_concat_default = True
