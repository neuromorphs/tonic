import tonic

project = "Tonic"
copyright = "2019-present, the neuromorphs of Telluride"
author = "Gregor Lenz"

master_doc = "index"

extensions = [
    "autoapi.extension",
    "myst_nb",
    "pbr.sphinxext",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

sphinx_gallery_conf = {
    "examples_dirs": "gallery/",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "backreferences_dir": None,
    "matplotlib_animations": True,
    "doc_module": ("tonic",),
    "download_all_examples": False,
    "ignore_pattern": r"utils\.py",
}

autodoc_typehints = "both"
autoapi_type = "python"
autoapi_dirs = ["../tonic"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST settings
# nb_execution_mode = "off"
nb_execution_timeout = 300
nb_execution_show_tb = True
nb_execution_excludepatterns = ["large_datasets.ipynb"]
suppress_warnings = ["myst.header"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "auto_examples/**.ipynb",
    "auto_examples/**.py",
    "auto_examples/**.md5",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".ipynb_checkpoints",
    "README.rst",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = tonic.__version__
html_logo = "_static/tonic-logo-black-bg.png"
html_favicon = "_static/tonic_favicon.png"
html_show_sourcelink = True
html_sourcelink_suffix = ""

html_theme_options = {
    "repository_url": "https://github.com/neuromorphs/tonic",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "develop",
    "path_to_docs": "docs",
    "use_fullscreen_button": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
