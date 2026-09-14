Installation
============

Recommended: pixi
*****************
``gEconpy`` is published on conda-forge, and the recommended way to install it is with `pixi <https://pixi.sh>`_.
From inside a pixi workspace, or a new one:

.. code-block:: bash

    pixi init my-dsge-project
    cd my-dsge-project
    pixi add geconpy
    pixi shell

pixi resolves ``gEconpy`` and `pytensor <https://pytensor.readthedocs.io/en/latest/>`_ from conda-forge, which ships
pytensor with a C compiler and a BLAS library already configured, and records the result in a lock file.


conda
*****
The same conda-forge package installs into a conda environment:

.. code-block:: bash

    conda create -n geconpy -c conda-forge python=3.12 geconpy
    conda activate geconpy


pip
***
``gEconpy`` is also on PyPI. Install pytensor from conda-forge first, then ``gEconpy`` with pip:

.. code-block:: bash

    conda create -n geconpy -c conda-forge python=3.12 pytensor
    conda activate geconpy
    pip install gEconpy

Installing straight from PyPI into a plain virtual environment also works, but pytensor then needs a C compiler on the
system path to compile models.


Development Installation
************************
The repository is itself a pixi workspace. From a clone, ``pixi install`` creates an environment with the runtime,
test, and documentation dependencies resolved from the committed lock file, with ``gEconpy`` installed in editable
mode. ``pixi shell`` activates it:

.. code-block:: bash

    git clone https://github.com/jessegrabowski/gEconpy.git
    cd gEconpy
    pixi install
    pixi shell

The workspace defines tasks for the common jobs, which run without activating a shell:

.. code-block:: bash

    pixi run test                # the test suite, extra pytest arguments pass through
    pixi run lint                # every pre-commit hook on every file
    pixi run docs-build          # the documentation, into docs/build/html

If you would rather use conda directly, ``conda_envs/environment_dev.yml`` is the same environment exported from the
pixi workspace. Its pip section installs ``gEconpy`` in editable mode from ``.``, so run the command from the
repository root:

.. code-block:: bash

    conda env create -f conda_envs/environment_dev.yml
    conda activate geconpy-dev
