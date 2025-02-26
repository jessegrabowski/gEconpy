Introduction
============
A collection of tools for working with DSGE models in python, inspired by the fantastic R package gEcon, http://gecon.r-forge.r-project.org/.

Like gEcon, gEconpy solves first order conditions automatically, helping the researcher avoid math errors while facilitating rapid prototyping of models. By working in the optimization problem space rather than the FoC space, modifications to the model are much simpler. Adding an additional term to the utility function, for example, requires modifying only 2-3 lines of code, whereas in FoC space it may require re-solving the entire model by hand.

gEconpy uses the GCN file originally created for the gEcon package. gEcon GCN files are fully compatable with gEconpy, and includes all the great features of GCN files, including:

* Automatically solve first order conditions
* Users can include steady-state values in equations without explictly solving for them by hand first!
* Users can declare "calibrated parameters", requesting a parameter value be found to induce a specific steady-state relationship

gEconpy is still in an unfinished alpha state, but I encourage anyone interested in DSGE modeling to give it a try and and report any bugs you might find.


Quick Setup
===========
gEconpy is available on PyPi, and can be installed with pip:

.. code-block:: bash

    pip install gEconpy


For more detailed installation instructions, see the :doc:`installation guide <install>`.


Citation
========

If you use gEconpy in your research, please cite the package using the following BibTeX entry:

.. code-block:: bibtex

   @software{gEconpy,
     author = {Jesse Grabowski},
     title = {gEconpy: A collection of tools for working with DSGE models in python},
     url = {https://github.com/jessegrabowski/gEconpy},
     version = {1.2.0}}

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   get_started/index
   user_guide/index
   examples/gallery
   api
   dev/index
   release/index
