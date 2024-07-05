Installation
============


Recommended Installation Method
*******************************
You can use the ``environment.yaml`` file provided in the repository to create a new conda environment with all the
dependencies installed:

.. code-block:: bash

    conda env create -f environment.yaml
    conda activate gEconpy


Other Methods
*************
``gEconpy`` is available on PyPI and can be installed using pip:

.. code-block:: bash

    pip install gEconpy

This command will install the package and all its dependencies. Is is **strongly** recommended that you create a
virtual environment before installing the package. ``gEconpy`` depends on `pytensor <https://pytensor.readthedocs.io/en/latest/>`_,
a package that requires a C compiler to be installed on your system. Therefore, it is recommended that you first create a virtual environment with
pytensor, then install ``gEconpy`` in that environment:

.. code-block:: bash

    conda create -n geconpy "python=3.12" pip pytensor
    conda activate geconpy
    pip install gEconpy


This will ensure that all dependencies are correctly installed and that the package will work as expected.
