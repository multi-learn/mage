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
repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(repo_path)
# print(os.path.join(repo_path, "multiview_generator",  "base"))
# quit()

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.join(repo_path, "multiview_generator'"))
sys.path.insert(0, repo_path)

# -- Project information -----------------------------------------------------

project = 'MAGE'
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
                'sphinx_rtd_theme',
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
                "autoapi.extension",
              'nbsphinx',
              "nbsphinx_link"
               # 'm2r'
              ]

autoapi_type = 'python'
autoapi_dirs = [os.path.join(repo_path, "multiview_generator",""),]
autoapi_options = ["members", "show-module-summary", 'undoc-members']
autoapi_ignore = ["*tests*"]
autoapi_keep_files = False
autoapi_add_toctree_entry = False
add_module_names = False
autoapi_template_dir = os.path.join(repo_path, "docs", "source", "templates_autoapi")
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', 'templates_autoapi']


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
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static',]

rst_prolog = """
.. role:: python(code)
    :language: python

.. role :: yaml(code)
    :language: yaml

.. |gene| replace:: MAGE

.. |gene_f| replace:: Multi-view Artificial Generation Engine

.. |HPO| replace:: hyper-parameters optimization
"""

extlinks = {'base_source': (
'https://gitlab.lis-lab.fr/dev/multiview_generator',
"base_source"),
            'base_doc': (
            'https://dev.pages.lis-lab.fr/multiview_generator/', 'base_doc'),
            'summit':('https://gitlab.lis-lab.fr/baptiste.bauvin/summit', 'summit')}

html_js_files = [
    'plotly_js.js',
]