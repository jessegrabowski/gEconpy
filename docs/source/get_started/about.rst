About gEconpy
=============

gEconpy is a Python reimplementation of the ideas in `gEcon <http://gecon.r-forge.r-project.org/>`_, the R package
by Grzegorz Klima, Karol Podemski and Kaja Retkiewicz-Wijtiwiak. Like gEcon, it reads a model written as a set of
optimization problems and derives the first-order conditions itself, so a change to a utility function is a change
to one line of the model file, and the conditions that follow from it are worked out again automatically. GCN files
written for gEcon parse without changes.

gEconpy differs from gEcon in the stack underneath. It does the symbolic work with SymPy, compiles the model into a
PyTensor graph, and runs estimation through PyMC and the pymc-extras state space module, so a model's likelihood is
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
