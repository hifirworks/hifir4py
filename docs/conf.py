# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

import datetime
import hifir4py

project = "hifir4py"
copyright = "2019-{}, NumGeom Group at Stony Brook University".format(datetime.date.today().year)
author = "Qiao Chen"

# The short X.Y version
version = hifir4py.__version__
# The full version, including alpha/beta/rc tags
release = hifir4py.__version__

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    # "numpydoc",
]

intersphinx_mapping = {
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_rtd_theme"
# html_theme = "nature"
html_theme = "sizzle"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {"github_url": "https://github.com/hifirworks/hifir4py"}

# html_context = {
#     "display_github": True,
#     "github_user": "hifirworks",
#     "github_repo": "hifir4py",
#     "github_version": "main/docs/",
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}
html_domain_indices = True

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
# htmlhelp_basename = "PyCpldoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
    "papersize": "a4paper",
    "pointsize": "11pt",
    # 'preamble': '',
    "figure_align": "htbp",
    "extraclassoptions": ",openany,oneside",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [(master_doc, "hifir4py.tex", "hifir4py Documentation", author, "manual")]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "hifir4py", "hifir4py Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "hifir4py",
        "hifir4py Documentation",
        author,
        "hifir4py",
        "One line description of project.",
        "Miscellaneous",
    )
]

# -- Extension configuration -------------------------------------------------
todo_include_todos = True
# numpydoc_show_class_members = False
add_module_names = False  # remove module names. i.e. module.foobar
napoleon_google_docstring = False
napoleon_numpy_docstring = True
autosummary_generate = True
