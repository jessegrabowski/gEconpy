About gEconpy
=============

gEconpy is a Python reimplementation of the ideas in `gEcon <http://gecon.r-forge.r-project.org/>`_, the R package
by Grzegorz Klima, Karol Podemski and Kaja Retkiewicz-Wijtiwiak. Like gEcon, it reads a model written as a set of
optimization problems and derives the first-order conditions itself, so a change to a utility function is a change
to one line of the model file, with no re-derivation by hand. GCN files written for gEcon parse without
changes.

Where gEconpy departs from gEcon is the stack underneath. Symbolic work is done with SymPy, the compiled model is a
PyTensor graph, and estimation runs through PyMC and the pymc-extras state space module, so a model's likelihood is
differentiable and can be sampled with gradient-based samplers.

gEconpy is developed by Jesse Grabowski. Issues and pull requests are welcome on
`GitHub <https://github.com/jessegrabowski/gEconpy>`_.

Citing
------

.. code-block:: bibtex

   @software{gEconpy,
     author = {Jesse Grabowski},
     title = {gEconpy: A collection of tools for working with DSGE models in python},
     url = {https://github.com/jessegrabowski/gEconpy}}
