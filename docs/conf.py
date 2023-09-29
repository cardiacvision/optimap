# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import inspect

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

import jupytext
import jupytext.config

try:
    from optimap import __version__ as release
except ImportError:
    release = "unknown"


# -- Convert tutorials -------------------------------------------------------
# convert notebooks to python files for download
def convert_notebooks():
    tutorials = Path(__file__).parent / "tutorials"
    for path in tutorials.glob("*.ipynb"):
        nb = jupytext.read(path)

        remove_cells_for_tags = ["remove-input"]

        def f(cell):
            for tag in remove_cells_for_tags:
                if tag in cell.metadata.get("tags", []):
                    return False
            return True
        nb['cells'] = list(filter(f, nb.cells))

        config = jupytext.config.JupytextConfiguration()
        config.notebook_metadata_filter = "-all"
        config.cell_metadata_filter = "-all"

        output_folder = path.parent / "converted"
        output_folder.mkdir(exist_ok=True)
        jupytext.write(nb, output_folder / f'{path.stem}.py', fmt="py:percent", config=config)
        jupytext.write(nb, output_folder / f'{path.stem}.ipynb', fmt="ipynb", config=config)
convert_notebooks()

# -- Project information -----------------------------------------------------

project = 'optimap'
copyright = '2023, Jan Lebert, Jan Christoph'
author = 'Jan Lebert, Jan Christoph'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_codeautolink",
    'sphinxcontrib.bibtex',
    "sphinxcontrib.video",
    "sphinx.ext.viewcode",
    "myst_nb",
]
myst_enable_extensions = [
    "dollarmath"
]

# API settings
autodoc_default_options = {
    "members": True,
    "show-inheritance": False,
    # "undoc-members": True,
    # "imported-members": True
}
autodoc_member_order = 'bysource'
autosummary_generate = True

codeautolink_global_preface = "import optimap as om"

nb_execution_mode = "cache"
nb_execution_timeout = 600  # seconds
nb_execution_excludepatterns = ['tutorials/converted/*']

# add_module_names = False
# napoleon_google_docstring = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = False
napoleon_numpy_docstring = True
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = False
# napoleon_use_rtype = False
# numpydoc_show_class_members = False

bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'unsrtalpha'
bibtex_reference_style = 'author_year'

# Intersphinx settings
intersphinx_mapping = {
    "ipywidgets": ("https://ipywidgets.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    # "monochrome": ("https://monochrome.readthedocs.io/en/latest/", None),  # TODO: fix
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tutorials/converted']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_copy_source = True  # needed for download notebook button
# html_css_files = [
#     "custom.css",
# ]
html_title = "optimap"
html_theme = "furo"

master_doc = "index"
# thebe_config = {
#     "repository_url": html_theme_options["repository_url"],
#     "repository_branch": html_theme_options["repository_branch"],
# }
html_theme_options = {
    "source_repository": "https://github.com/cardiacvision/optimap/",
    "source_branch": "main",
    "source_directory": "docs/",
}


# based on pandas/doc/source/conf.py
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname("../optimap"))

    return f"https://github.com/cardiacvision/optimap/blob/main/optimap/{fn}{linespec}"  # noqa