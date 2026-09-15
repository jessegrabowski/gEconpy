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
    "sphinx_codeautolink",
    "generate_gallery",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "numpydoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
]

# Make autosectionlabel use document path as prefix to avoid duplicate labels
autosectionlabel_prefix_document = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "**/.ipynb_checkpoints",
    "examples/GCN Files",
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
add_module_names = False
autoclass_content = "class"

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
    "to",
    "mapping",
    "expression",
    "tuples",
    "transform",
    "symbol",
    "nested",
    "a",
    "Matplotlib",
    "pymc",
    "pandas",
    "arviz",
    "preliz",
    "matrix",
    "expressions",
    "distribution",
    "and",
    # Real classes whose projects publish no intersphinx entry for them: pymc's Transform base class,
    # IPython's HTML, pyparsing's ParseBaseException and functools' lru_cache CacheInfo.
    "Transform",
    "HTML",
    "ParseBaseException",
    "CacheInfo",
}

# numpydoc_xref_param_type cross-references every word of a type field, so a class named bare or by its
# import alias needs its documented path here. Third-party targets resolve through intersphinx.
numpydoc_xref_aliases = {
    "SymbolDictionary": "gEconpy.classes.containers.SymbolDictionary",
    "SteadyStateResults": "gEconpy.classes.containers.SteadyStateResults",
    "TimeAwareSymbol": "gEconpy.classes.time_aware_symbol.TimeAwareSymbol",
    "Block": "gEconpy.model.block.basic.Block",
    "Model": "gEconpy.model.model.Model",
    "DSGEStateSpace": "gEconpy.model.statespace.DSGEStateSpace",
    "Variable": "gEconpy.parser.ast.nodes.Variable",
    "Parameter": "gEconpy.parser.ast.nodes.Parameter",
    "Tag": "gEconpy.parser.ast.nodes.Tag",
    "TimeIndex": "gEconpy.parser.ast.nodes.TimeIndex",
    "Node": "gEconpy.parser.ast.nodes.Node",
    "GCNModel": "gEconpy.parser.ast.nodes.GCNModel",
    "GCNBlock": "gEconpy.parser.ast.nodes.GCNBlock",
    "GCNEquation": "gEconpy.parser.ast.nodes.GCNEquation",
    "GCNDistribution": "gEconpy.parser.ast.nodes.GCNDistribution",
    "ErrorCollector": "gEconpy.parser.errors.ErrorCollector",
    "ParseLocation": "gEconpy.parser.errors.ParseLocation",
    "GCNParseError": "gEconpy.parser.errors.GCNParseError",
    "GCNGrammarError": "gEconpy.parser.errors.GCNGrammarError",
    "GCNSemanticError": "gEconpy.parser.errors.GCNSemanticError",
    "GCNErrorCollection": "gEconpy.parser.errors.GCNErrorCollection",
    "ErrorCode": "gEconpy.parser.error_catalog.ErrorCode",
    "ErrorInfo": "gEconpy.parser.error_catalog.ErrorInfo",
    "ParseResult": "gEconpy.parser.preprocessor.ParseResult",
    "RootSolver": "gEconpy.solvers.sparse_root.base.RootSolver",
    "DirectionStrategy": "gEconpy.solvers.sparse_root.direction.DirectionStrategy",
    "GlobalizationStrategy": "gEconpy.solvers.sparse_root.globalization.GlobalizationStrategy",
    "Distribution": "preliz.distributions.distributions.Distribution",
    "TensorVariable": "pytensor.tensor.TensorVariable",
    "Apply": "pytensor.graph.basic.Apply",
    "Mode": "pytensor.compile.mode.Mode",
    "pd.DataFrame": "pandas.DataFrame",
    "DataFrame": "pandas.DataFrame",
    "pd.DatetimeIndex": "pandas.DatetimeIndex",
    "DatetimeIndex": "pandas.DatetimeIndex",
    "xr.DataArray": "xarray.DataArray",
    "DataArray": "xarray.DataArray",
    "xr.Dataset": "xarray.Dataset",
    "Dataset": "xarray.Dataset",
    "xr.DataTree": "xarray.DataTree",
    "DataTree": "xarray.DataTree",
    "InferenceData": "arviz.InferenceData",
    "sp.Expr": "sympy.core.expr.Expr",
    "sympy.Expr": "sympy.core.expr.Expr",
    "Expr": "sympy.core.expr.Expr",
    "sp.Symbol": "sympy.core.symbol.Symbol",
    "sympy.Symbol": "sympy.core.symbol.Symbol",
    "Symbol": "sympy.core.symbol.Symbol",
    "sp.Eq": "sympy.core.relational.Eq",
    "sympy.Eq": "sympy.core.relational.Eq",
    "Eq": "sympy.core.relational.Eq",
    "sp.Basic": "sympy.core.basic.Basic",
    "MutableDenseMatrix": "sympy.matrices.dense.MutableDenseMatrix",
    "Matrix": "sympy.matrices.dense.Matrix",
    "Figure": "matplotlib.figure.Figure",
    "plt.Figure": "matplotlib.figure.Figure",
    "Colormap": "matplotlib.colors.Colormap",
    "Axes": "matplotlib.axes.Axes",
    "AxesImage": "matplotlib.image.AxesImage",
    "Colorbar": "matplotlib.colorbar.Colorbar",
    "Formatter": "matplotlib.ticker.Formatter",
    "GridSpec": "matplotlib.gridspec.GridSpec",
    "ArrayLike": "numpy.typing.ArrayLike",
    "np.random.Generator": "numpy.random.Generator",
    "Path": "pathlib.Path",
    "pm.Model": "pymc.model.core.Model",
    "OptimizeResult": "scipy.optimize.OptimizeResult",
    "sparse.csc_matrix": "scipy.sparse.csc_matrix",
}

# A role naming a symbol that moved or was deleted otherwise renders as plain text with a green build.
nitpicky = True

# Targets no docstring can make resolve: projects whose inventory does not carry the object, and two
# pytensor gradient helpers the pytensor inventory omits.
nitpick_ignore_regex = [
    ("py:.*", r"arviz\..*"),
    ("py:.*", r"preliz\..*"),
    ("py:.*", r"sympytensor\..*"),
    ("py:func", r"pytensor\.gradient\.grad_(undefined|not_implemented)"),
]

# Op subclasses override ``perform`` without a docstring, so autodoc inherits pytensor's, whose roles are
# written relative to the pytensor namespace and cannot resolve from here.
nitpick_ignore = [
    ("py:attr", "node.inputs"),
    ("py:attr", "node.outputs"),
    ("py:meth", "Op.perform"),
]

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


# -- Intersphinx Mapping -------------------------------------------------
intersphinx_mapping = {
    "arviz": ("https://python.arviz.org/en/latest/", None),
    "pytensor": ("https://pytensor.readthedocs.io/en/latest/", None),
    "pmx": ("https://www.pymc.io/projects/extras/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "myst": ("https://myst-parser.readthedocs.io/en/latest", None),
    "myst-nb": ("https://myst-nb.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "preliz": ("https://preliz.readthedocs.io/en/latest/", None),
}
