# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = "Palmprint Scanner"
author = "L. Basedow, L. Gillner, J. Prothmann, C. Werner"
copyright = "2021, Basedow, Gillner, Prothmann, Werner"
release = "1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]

language = "de"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "white",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
html_static_path = ["_static"]
html_logo = "./_static/icons8-palm-scan.png"
html_search_language = "de"
html_title = "Palmprint Scanner Dokumentation"
html_short_title = "PpS Doc"

# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../"))


def skip(app, what, name, obj, would_skip, options):
    if name in ("__init__",):
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


extensions.append("sphinx_autodoc_typehints")
