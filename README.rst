.. |pipeline| image:: https://gitlab.lis-lab.fr/dev/multiview_generator/badges/master/pipeline.svg
    :alt: Pipeline status

.. |license| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: License: New BSD

.. |coverage| image:: https://gitlab.lis-lab.fr/dev/multiview_generator/badges/master/coverage.svg
    :target: http://dev.pages.lis-lab.fr/multiview_generator/coverage/index.html
    :alt: Coverage

|pipeline| |license| |coverage|

MAGE : Multi-view Artificial Generation Engine
==============================================

.. image:: docs/source/_static/mage-small.png

This package aims at generating customized mutli-view datasets to facilitate the
development of new multi-view algorithms and their testing on simulated data
representing specific tasks.

Getting started
---------------

This code has been originally developed on Ubuntu, but if the compatibility
with Mac or Windows is mandatory for you, contact us so we adapt it.

+----------+-------------------+
| Platform | Last positive test|
+==========+===================+
|   Linux  |  |pipeline|       |
+----------+-------------------+
| Mac      | Not verified yet  |
+----------+-------------------+
| Windows  | Not verified yet  |
+----------+-------------------+

Prerequisites
<<<<<<<<<<<<<

To be able to use this project, you'll need :

* `Python 3 <https://docs.python.org/3/>`_

And the following python modules will be automatically installed  :

* `numpy <http://www.numpy.org/>`_, `scipy <https://scipy.org/>`_,
* `matplotlib <http://matplotlib.org/>`_ - Used to plot results,
* `sklearn <http://scikit-learn.org/stable/>`_ - Used for the monoview classifiers,
* `h5py <https://www.h5py.org>`_ - Used to generate HDF5 datasets on hard drive and use them to spare RAM,
* `pandas <https://pandas.pydata.org/>`_ - Used to manipulate data efficiently,
* `docutils <https://pypi.org/project/docutils/>`_ - Used to generate documentation,
* `pyyaml <https://pypi.org/project/PyYAML/>`_ - Used to read the config files,
* `plotly <https://plot.ly/>`_ - Used to generate interactive HTML visuals,
* `tabulate <https://pypi.org/project/tabulate/>`_ - Used to generated the confusion matrix,
* `jupyter <https://jupyter.org/>`_ - Used for the tutorials


Installing
<<<<<<<<<<

Once you cloned the project from the `Github repository <https://github.com/multi-learn/mage>`_, you just have to use :

.. code:: bash

    cd path/to/multiview_generator/
    pip3 install -e .


In the `multiview_generator` directory to install MAGE and its dependencies.


Running the tests
<<<<<<<<<<<<<<<<<

To run the test suite of MAGE, run :

.. code:: bash

    cd path/to/multiview_generator
    pip install -e .[dev]
    pytest

The coverage report is automatically generated and stored in the ``htmlcov/`` directory

Building the documentation
<<<<<<<<<<<<<<<<<<<<<<<<<<

To locally build the `documentation <https://multi-learn.github.io/mage/>`_ run :

.. code:: bash

    cd path/to/multiview_generator
    pip install -e .[doc]
    python setup.py build_sphinx

The locally built html files will be stored in ``path/to/multiview_generator/build/sphinx/html``

Authors
-------

* **Baptiste BAUVIN**
* **Dominique BENIELLI**
* **Sokol Koço**