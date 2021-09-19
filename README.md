# gEcon.py
A collection of tools for working with DSGE models in python, inspired by the fantastic R package gEcon, http://gecon.r-forge.r-project.org/.

In particular, this (admittedly disorganized) code can parse a GCN file, automatrically derive first-order conditions, and solve for the system's steady-state. It uses a mixture of Sympy, numpy, and scipy to accomplish all of this.

Once a steady-state is computed, the system can be written into a .mod file for use in Dynare, or the steady-state solution can be written into a .R file for use with gEcon. There is currently no support for actually log-linearizing the system, solving for the policy function, or simulating the model in native python. 

That's on the ol' to-do list, though.
