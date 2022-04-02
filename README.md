# gEcon.py
A collection of tools for working with DSGE models in python, inspired by the fantastic R package gEcon, http://gecon.r-forge.r-project.org/.

Like gEcon, gEcon.py solves first order conditions automatically, helping the researcher avoid math errors while facilitating rapid prototyping of models. By working in the optimization problem space rather than the FoC space, modifications to the model are much simpler. Adding an additional term to the utility function, for example, requires modifying only 2-3 lines of code, whereas in FoC space it would require re-solving the entire model by hand.

gEcon.py uses the GCN file originally created for the gEcon package. gEcon GCN files are fully compatable with gEcon.py, and includes all the great features of GCN files, including:
* Automatically solve first order conditions
* Users can include steady-state values in equations without explictly solving for them by hand first!
* Users can declare "calibrated parameters", requesting a parameter value be found to induce a specific steady-state relationship

In addition, gEcon.py adds several additional features to the GCN file, including:

* Define priors on parameters directly in the GCN file, using a natural alpha ~ N(mean=0.35, sd=0.1) notation
* Provide full or partial steady state solution in a special STEADY_STATE block, increasing the efficency of the numerical solver and speeding up MCMC sampling (in theory, MCMC isn't implemented yet!)

Finally, since Dynare remains the gold standard for DSGE model implementation, gEcon.py can convert a solved model to a Dynare .mod file!
* But no efforts are made to provide symbolic solutions to the steady-state! If you include steady state values in your equations and do not provide a symbolic solution in the STEADY_STATE block, the .mod file will not work "out of the box".
* This is also true of claibrated equations!

gEcon.py is still in an unfinished alpha state, but I encourage anyone interested in DSGE modeling to give it a try and and report any bugs you might find.
