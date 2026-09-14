Package Overview
================

gEconpy takes a DSGE model written in a GCN file and turns it into a :class:`~gEconpy.model.model.Model` object that
can find its own steady state, be solved by perturbation or under perfect foresight, be simulated, and be estimated
with PyMC. The user guide and the example gallery show these pieces in use.

A GCN file
----------

A GCN file lists the agents in the model as blocks. Each block gives the agent's objective, controls and constraints,
and gEconpy derives the first-order conditions itself. A block for an exogenous process holds the law of motion and
the shock. Parameters get fixed values, priors with starting values, or calibrating equations, and an optional
``STEADY_STATE`` block gives whatever part of the steady state you can write down. The format and every component are
covered in :doc:`/user_guide/gcn_files/syntax`, using the RBC model shipped with the package.

From a file to a model
----------------------

:func:`~gEconpy.model.build.model_from_gcn` parses a file, derives the first-order conditions, and compiles the
system of equations, its Jacobians, and the steady-state residuals into callable functions. The resulting
:class:`~gEconpy.model.model.Model` keeps the symbolic system alongside the compiled functions, so the same model can
be solved and re-solved under new parameter values without recompiling.

Steady state and solution
-------------------------

:meth:`~gEconpy.model.model.Model.steady_state` combines the analytical relationships from the GCN file with a
numerical solver for the rest, and :meth:`~gEconpy.model.model.Model.solve_model` linearizes around it and finds the
policy function by cycle reduction or gensys. :func:`~gEconpy.model.simulate.impulse_response_function` and
:func:`~gEconpy.model.simulate.simulate` use that solution, and the functions in
:doc:`gEconpy.model.statistics </api/gEconpy.model.statistics>` compute stationary covariances, autocorrelations, and solvability diagnostics.
:func:`~gEconpy.model.perfect_foresight.solve.solve_perfect_foresight` solves the nonlinear model for a known shock
path without linearizing it.

Estimation
----------

:func:`~gEconpy.model.build.statespace_from_gcn` builds a :class:`~gEconpy.model.statespace.DSGEStateSpace`, a
pymc-extras state-space model whose parameters carry the priors declared in the GCN file. Configuring it with the
observed variables and building its graph inside a PyMC model gives a Kalman-filter likelihood that any PyMC sampler
can use. The estimation notebooks in the gallery walk through this end to end.
