***********
Sparse Root
***********

The :mod:`~gEconpy.solvers.sparse_root` package provides composable solvers for
nonlinear systems with sparse Jacobians. It is the primary solver backend for
perfect-foresight simulation.

All solvers are used through a single entry point,
:func:`~gEconpy.solvers.sparse_root.sparse_root`, which accepts a solver
instance controlling the algorithm.

.. code-block:: python

    from gEconpy.solvers.sparse_root import sparse_root, NewtonArmijo

    result = sparse_root(fun, x0, solver=NewtonArmijo())


Entry Point
===========

.. autofunction:: gEconpy.solvers.sparse_root.sparse_root


Solvers
=======

Every solver is an object with ``init`` and ``step`` methods. The package ships
seven solvers in two families.


Line-search solvers
-------------------

Line-search solvers compose a *direction strategy* (how to choose a search
direction) with a *globalization strategy* (how to choose a step length along
that direction). The four named constructors below return a
:class:`~gEconpy.solvers.sparse_root.line_search.LineSearchSolver` with
sensible defaults; pass ``direction`` or ``globalization`` to override either
component.

.. code-block:: python

    from gEconpy.solvers.sparse_root import NewtonArmijo
    from gEconpy.solvers.sparse_root.direction import KrylovDirection
    from gEconpy.solvers.sparse_root.globalization import NonmonotoneBacktracking

    # Mix and match: Krylov direction with nonmonotone line search
    solver = NewtonArmijo(
        direction=KrylovDirection(krylov_method="gmres"),
        globalization=NonmonotoneBacktracking(memory=10),
    )

.. autofunction:: gEconpy.solvers.sparse_root.line_search.NewtonArmijo

.. autofunction:: gEconpy.solvers.sparse_root.line_search.Chord

.. autofunction:: gEconpy.solvers.sparse_root.line_search.InexactNewtonKrylov

.. autofunction:: gEconpy.solvers.sparse_root.line_search.NewtonNonmonotone

.. autoclass:: gEconpy.solvers.sparse_root.line_search.LineSearchSolver
   :members: init, step


Trust-region solvers
--------------------

Trust-region solvers manage their own step-acceptance logic via an adaptive
trust-region radius, so they do not use a separate globalization strategy.

.. autoclass:: gEconpy.solvers.sparse_root.dogleg.SparseDogleg
   :members: init, step

.. autoclass:: gEconpy.solvers.sparse_root.gauss_newton.GaussNewtonTrustRegion
   :members: init, step

.. autoclass:: gEconpy.solvers.sparse_root.levenberg_marquardt.LevenbergMarquardt
   :members: init, step


Direction Strategies
====================

Direction strategies compute the search direction at each iteration. They are
passed to line-search solvers via the ``direction`` argument.

.. autoclass:: gEconpy.solvers.sparse_root.direction.NewtonDirection
   :members: compute

.. autoclass:: gEconpy.solvers.sparse_root.direction.ChordDirection
   :members: compute, reset

.. autoclass:: gEconpy.solvers.sparse_root.direction.KrylovDirection
   :members: compute, reset


Globalization Strategies
========================

Globalization strategies control step-length selection. They are passed to
line-search solvers via the ``globalization`` argument.

.. autoclass:: gEconpy.solvers.sparse_root.globalization.ArmijoBacktracking
   :members: search

.. autoclass:: gEconpy.solvers.sparse_root.globalization.NonmonotoneBacktracking
   :members: search
