Configuration
=============

``gEconpy.rcParams`` holds the defaults that model building, the solvers, and estimation use when no argument is
given.
An argument passed to a function always takes precedence over the value in ``rcParams``.

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Used by
   * - ``build.verbose``
     - ``True``
     - :func:`~gEconpy.model.build.model_from_gcn` and :func:`~gEconpy.model.build.statespace_from_gcn`
   * - ``build.mode``
     - ``None``
     - :func:`~gEconpy.model.build.model_from_gcn` and :func:`~gEconpy.model.build.statespace_from_gcn`
   * - ``build.on_unused_parameters``
     - ``"raise"``
     - :func:`~gEconpy.model.build.model_from_gcn` and :func:`~gEconpy.model.build.statespace_from_gcn`
   * - ``build.show_errors``
     - ``True``
     - :func:`~gEconpy.model.build.model_from_gcn` and :func:`~gEconpy.model.build.statespace_from_gcn`
   * - ``steady_state.tol``
     - ``1e-8``
     - Absolute residual tolerance of :meth:`~gEconpy.model.model.Model.steady_state`
   * - ``steady_state.rtol``
     - ``1e-10``
     - Tolerance relative to the sum of an equation's term magnitudes, added to ``steady_state.tol``
   * - ``steady_state.grad_tol``
     - ``1e-6``
     - Gradient tolerance of a numerical steady-state solve
   * - ``sparse_root.tol``
     - ``1e-10``
     - :func:`~gEconpy.solvers.sparse_root.sparse_root.sparse_root` and the perfect-foresight solvers
   * - ``sparse_root.maxiter``
     - ``1000``
     - :func:`~gEconpy.solvers.sparse_root.sparse_root.sparse_root` and the perfect-foresight solvers
   * - ``perturbation.tol``
     - ``1e-8``
     - :meth:`~gEconpy.model.model.Model.solve_model` and
       :func:`~gEconpy.model.statistics.perturbation_diagnostics.solvability_check`
   * - ``statespace.solver``
     - ``"gensys"``
     - :meth:`~gEconpy.model.statespace.DSGEStateSpace.configure`
   * - ``statespace.max_iter``
     - ``50``
     - :meth:`~gEconpy.model.statespace.DSGEStateSpace.configure`
   * - ``statespace.tol``
     - ``1e-6``
     - :meth:`~gEconpy.model.statespace.DSGEStateSpace.configure`
   * - ``statespace.use_adjoint_gradients``
     - ``True``
     - :meth:`~gEconpy.model.statespace.DSGEStateSpace.configure`
   * - ``statespace.adjoint_method``
     - ``"doubling"``
     - :meth:`~gEconpy.model.statespace.DSGEStateSpace.configure`
   * - ``statespace.use_direct_lyapunov``
     - ``False``
     - :meth:`~gEconpy.model.statespace.DSGEStateSpace.configure`
   * - ``statespace.solver_tol``
     - ``1e-8``
     - :meth:`~gEconpy.model.statespace.DSGEStateSpace.build_statespace_graph`

Setting a value checks the key and the value, so a misspelled key or a negative tolerance raises immediately.
:func:`~gEconpy.rcparams.rc_context` changes settings for the duration of a ``with`` block and restores them on exit:

.. code-block:: python

    import gEconpy as ge

    ge.rcParams["steady_state.rtol"] = 1e-8

    with ge.rc_context({"steady_state.rtol": 1e-8, "steady_state.tol": 1e-6}):
        steady_state = model.steady_state()

At import, gEconpy reads a file named ``geconpyrc`` from the working directory or, failing that, from
``$XDG_CONFIG_HOME/geconpy/``, where ``XDG_CONFIG_HOME`` defaults to ``~/.config``. Each line holds ``key : value``,
and ``#`` starts a comment:

.. code-block:: text

    # ~/.config/geconpy/geconpyrc
    build.verbose : False
    steady_state.rtol : 1e-8
    sparse_root.maxiter : 500
    statespace.adjoint_method : qz

An invalid line makes the import fail with an error naming the file and line.
