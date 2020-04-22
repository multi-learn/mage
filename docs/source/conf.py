# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../multiview_generator'))

# -- Project information -----------------------------------------------------

project = 'Mulitivew Generator'
copyright = '2020, Baptiste Bauvin'
author = 'Baptiste Bauvin'

# The full version, including alpha/beta/rc tags
release = '0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.extlinks',
#              'sphinx.ext.doctest',
#              'sphinx.ext.intersphinx',
#              'sphinx.ext.todo',
#              'nbsphinx',
#              'sphinx.ext.coverage',
               'sphinx.ext.imgmath',
#              'sphinx.ext.mathjax',
#              'sphinx.ext.ifconfig',
#              'sphinx.ext.viewcode',
#              'sphinx.ext.githubpages',
               'sphinx.ext.napoleon',
              'nbsphinx',
              "nbsphinx_link"
               # 'm2r'
              ]

source_suffix = ['.rst', '.md', '.ipynb', ".nblink"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'groundwork'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

rst_prolog = """
.. role:: python(code)
    :language: python

.. role :: yaml(code)
    :language: yaml

.. |gene| replace:: SMuDGE

.. |gene_f| replace:: Supervised MUltimodal Dataset Generation Engine

.. |HPO| replace:: hyper-parameters optimization
"""

extlinks = {'base_source': (
'https://gitlab.lis-lab.fr/baptiste.bauvin/smudge/-/tree/master/',
"base_source"),
            'base_doc': (
            'http://baptiste.bauvin.pages.lis-lab.fr/smudge/', 'base_doc'),
            'summit':('https://gitlab.lis-lab.fr/baptiste.bauvin/summit', 'summit')}

html_js_files = [
    'plotly_js.js',
]