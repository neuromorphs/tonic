# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import tonic

# -- Project information -----------------------------------------------------

project = "Tonic"
copyright = "2019-2021, the neuromorphs of Telluride"
author = "Gregor Lenz"

version = ".".join(tonic.__version__.split("."))
release = tonic.__version__

master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_nb"]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST settings
jupyter_execute_notebooks = "off"
suppress_warnings = ["myst.header"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_book_theme"
html_logo = "_static/tonic-logo-black.png"
html_favicon = "_static/tonic_favicon.png"
html_show_sourcelink = True
html_sourcelink_suffix = ""

html_theme_options = {
    "logo_only": True,
    "repository_url": "https://github.com/neuromorphs/tonic",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_fullscreen_button": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
