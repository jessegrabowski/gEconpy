Solving a Model
===============

Once :func:`~gEconpy.model.build.model_from_gcn` has built a :class:`~gEconpy.model.model.Model`, solving it means
finding the steady state, then either a linear policy function around it or a nonlinear path away from it. This
page names the solvers and the choices each one exposes. The introductory notebooks in the
:doc:`example gallery </examples/gallery>` show them running.

Steady state
------------

:meth:`~gEconpy.model.model.Model.steady_state` returns a :class:`~gEconpy.classes.containers.SteadyStateResults`
mapping every variable to its steady-state value, with a ``success`` flag. Whatever the GCN file's
``STEADY_STATE`` block provides is used as given, and the remaining variables are found numerically. The ``how``
argument selects the method. The default ``"analytic"`` uses the block alone when it is complete and falls back to
a numerical solve otherwise, ``"root"`` solves the residual system directly, and ``"minimize"`` minimizes the
squared residuals, which is more forgiving of a poor starting point. Keyword arguments beyond those
are parameter overrides, so ``model.steady_state(beta=0.98)`` re-solves under a new value without rebuilding.

:func:`~gEconpy.model.statistics.validation.check_steady_state` reports which equations still have a residual when
a solve fails, which is the first thing to look at when the analytical block is suspect.

Perturbation
------------

:meth:`~gEconpy.model.model.Model.solve_model` linearizes the model around its steady state and returns the policy
matrices ``T`` and ``R`` of the linear state space representation :math:`x_t = T x_{t-1} + R \epsilon_t`. Two
solvers are available through the ``solver`` argument. ``"gensys"`` is Sims' QZ-decomposition method and reports
whether the solution exists and is unique. ``"cycle_reduction"`` iterates on the quadratic matrix equation and is
the method that is differentiable through PyTensor, which is why estimation uses its scan-based variant. Both
check the Blanchard-Kahn condition, and :func:`~gEconpy.model.perturbation.check_bk_condition` runs the check on
its own.

``log_linearize=True``, the default, linearizes in logs, so the policy function describes percentage deviations.
A variable with a negative or zero steady state cannot be log-linearized, and the ``assumptions`` block of the
GCN file is where a variable's sign is declared.

:func:`~gEconpy.model.statistics.perturbation_diagnostics.summarize_perturbation_solution` tabulates the solution,
and :func:`~gEconpy.model.statistics.perturbation_diagnostics.solvability_check` sweeps a set of parameter draws
and reports for each whether a steady state and a stable unique solution exist, which is the fastest way to find
the region of the parameter space where a model is well defined.

Moments and responses
---------------------

With ``T`` and ``R`` in hand, :func:`~gEconpy.model.simulate.impulse_response_function` and
:func:`~gEconpy.model.simulate.simulate` produce responses to a shock and simulated paths, and the functions in
:doc:`gEconpy.model.statistics </api/gEconpy.model.statistics>` compute the stationary covariance matrix and the
autocovariance and autocorrelation matrices at any number of lags. Each of these accepts the shock scale as a
single standard deviation, a dictionary per shock, or a full covariance matrix, and solves the model itself when
``T`` and ``R`` are not passed.

Perfect foresight
-----------------

:func:`~gEconpy.model.perfect_foresight.solve.solve_perfect_foresight` solves the nonlinear model for a known
sequence of shocks over ``T`` periods, stacking every period's equations into one sparse system and finding its
root. The result is exact for the nonlinear model, so it is the tool for large shocks, occasionally binding
constraints, and transitions between steady states. :func:`~gEconpy.model.perfect_foresight.solve.make_piecewise_x0`
builds a starting path that jumps between two steady states, which is the right guess for a permanent shock.

The stacked system is solved by :func:`~gEconpy.solvers.sparse_root.sparse_root.sparse_root`, which takes a fused
residual-and-Jacobian function and a solver object. The package ships two families of solvers.

Line-search solvers compose a direction strategy, which picks the search direction from the Jacobian, with a
globalization strategy, which picks the step length along it. Four named constructors return a
:class:`~gEconpy.solvers.sparse_root.line_search.LineSearchSolver` with a standard pairing, and each accepts
``direction`` and ``globalization`` to swap either part:

- :func:`~gEconpy.solvers.sparse_root.line_search.NewtonArmijo`: a full Newton direction with Armijo backtracking.
  This is the default solver.
- :func:`~gEconpy.solvers.sparse_root.line_search.Chord`: a Newton direction that reuses the factorized Jacobian
  for several steps, with Armijo backtracking.
- :func:`~gEconpy.solvers.sparse_root.line_search.InexactNewtonKrylov`: a Krylov-solved Newton direction, for
  systems too large to factorize, with Armijo backtracking.
- :func:`~gEconpy.solvers.sparse_root.line_search.NewtonNonmonotone`: a Newton direction with nonmonotone
  backtracking, which accepts occasional increases in the residual and escapes flat regions.

Trust-region solvers, :class:`~gEconpy.solvers.sparse_root.dogleg.SparseDogleg` and
:class:`~gEconpy.solvers.sparse_root.gauss_newton.GaussNewtonTrustRegion`, adapt a trust-region radius, and
:class:`~gEconpy.solvers.sparse_root.levenberg_marquardt.LevenbergMarquardt` adapts a damping parameter. None of
the three takes a globalization strategy.

Every solver is an object with ``init`` and ``step`` methods, so a new one is a class that implements those two.
Swapping a component is a keyword argument. This test replaces the direct sparse factorization inside a Newton
direction with GMRES:

.. literalinclude:: ../../../tests/solvers/sparse_root/test_line_search.py
   :language: python
   :pyobject: TestNewtonArmijoSpecific.test_custom_linear_solver
   :dedent: 4

The ``merit_fun`` argument of both backtracking strategies is the main performance lever for perfect foresight. A
line search evaluates the residual at every trial step, and the full function also builds the Jacobian. A merit
function that returns the residual alone lets the search skip the Jacobian until a step is accepted.
