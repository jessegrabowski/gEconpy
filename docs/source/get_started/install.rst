Installation
============


Recommended Installation Method
*******************************
The repository is a `pixi <https://pixi.sh>`_ workspace. From a clone, ``pixi install`` creates the development
environment with every runtime, test, and documentation dependency, and ``pixi shell`` activates it:

.. code-block:: bash

    pixi install
    pixi shell

If you would rather use conda directly, ``conda_envs/environment_dev.yml`` is the same environment exported from the
pixi workspace. Its pip section installs gEconpy in editable mode from ``.``, so run the command from the repository
root:

.. code-block:: bash

    conda env create -f conda_envs/environment_dev.yml
    conda activate geconpy-dev


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
