Solving a Model
===============

Once :func:`~gEconpy.model.build.model_from_gcn` has built a :class:`~gEconpy.model.model.Model`, solving it means
finding the steady state, then either a linear policy function around it or a nonlinear path away from it. This
page names the solvers and the choices each one exposes. The introductory notebooks in the
:doc:`example gallery </examples/gallery>` show them running.

Steady state
------------

:meth:`~gEconpy.model.model.Model.steady_state` returns a :class:`~gEconpy.classes.containers.SteadyStateResults`
mapping every variable to its steady-state value, with a ``success`` flag. gEconpy takes whatever the GCN file's
``Steady_State`` block provides as given and solves numerically for the remaining variables. The ``how`` argument
selects the method. The default ``"analytic"`` uses the block alone when it is complete and falls back to
a numerical solve otherwise, ``"root"`` solves the residual system directly, and ``"minimize"`` minimizes the
squared residuals, which is more forgiving of a poor starting point. Keyword arguments beyond those
are parameter overrides, so ``model.steady_state(beta=0.98)`` re-solves under a new value without rebuilding.

:func:`~gEconpy.model.statistics.validation.check_steady_state` reports which equations still have a residual when
a solve fails. Start there when the analytical block is suspect.

Perturbation
------------

:meth:`~gEconpy.model.model.Model.solve_model` linearizes the model around its steady state and returns the policy
matrices ``T`` and ``R`` of the linear state space representation :math:`x_t = T x_{t-1} + R \epsilon_t`. Two
solvers are available through the ``solver`` argument. ``"gensys"`` is Sims' QZ-decomposition method and reports
whether the solution exists and is unique. ``"cycle_reduction"`` iterates on the quadratic matrix equation and is
differentiable through PyTensor, so estimation uses its scan-based variant. Both check the Blanchard-Kahn
condition, and :func:`~gEconpy.model.perturbation.check_bk_condition` runs the check on its own.

``log_linearize=True``, the default, linearizes in logs, so the policy function describes percentage deviations.
A variable with a negative or zero steady state cannot be log-linearized. Declare a variable's sign in the
``assumptions`` block of the GCN file.

:func:`~gEconpy.model.statistics.perturbation_diagnostics.summarize_perturbation_solution` tabulates the solution,
and :func:`~gEconpy.model.statistics.perturbation_diagnostics.solvability_check` sweeps a set of parameter draws
and reports for each whether a steady state and a stable unique solution exist, which maps out the region of
the parameter space where a model is well defined.

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
root. The result is exact for the nonlinear model, so it handles large shocks, occasionally binding constraints,
and transitions between steady states. :func:`~gEconpy.model.perfect_foresight.solve.make_piecewise_x0`
builds a starting path that jumps between two steady states, which is the right guess for a permanent shock.

Before solving, each stacked equation is divided by the largest entry of its Jacobian row at the initial guess. A
model whose equations hold terms of very different sizes, such as a recursive wage-setting auxiliary of order
:math:`10^5` next to an Euler equation of order one, otherwise has its line search and convergence test dominated
by the large equation. The returned ``fun`` and ``jac`` are unscaled, but the convergence tolerance applies to the
scaled residuals, so the residual of an equation with terms of order :math:`10^5` can exceed the tolerance by about
that factor. In double precision such an equation cannot be solved to a much smaller absolute residual in any case.

The stacked system is solved by :func:`~gEconpy.solvers.sparse_root.sparse_root.sparse_root`, which takes a fused
residual-and-Jacobian function and a solver object. The package ships two families of solvers.

Line-search solvers compose a direction strategy, which picks the search direction from the Jacobian, with a
globalization strategy, which picks the step length along it. Four named constructors return a
:class:`~gEconpy.solvers.sparse_root.line_search.LineSearchSolver` with a standard pairing, and each accepts
``direction`` and ``globalization`` to swap either part:

- :func:`~gEconpy.solvers.sparse_root.line_search.newton_armijo`: a full Newton direction with Armijo backtracking.
  This is the default solver.
- :func:`~gEconpy.solvers.sparse_root.line_search.chord`: a Newton direction that reuses the factorized Jacobian
  for several steps, with Armijo backtracking. One factorization serves ``recompute_every`` steps, five by
  default, so a chord step costs a triangular solve rather than a factorization on four steps in five.
- :func:`~gEconpy.solvers.sparse_root.line_search.inexact_newton_krylov`: a Newton direction solved inexactly with
  GMRES, preconditioned by an LU factorization refreshed every five steps, with Armijo backtracking. It replaces
  most factorizations with cheaper preconditioned iterations, so it is faster than a direct solve on hard problems
  that take many Newton steps. It is slightly slower on easy ones, where its loose initial forcing term costs a few
  extra iterations. For a system too large to factorize at all, pass a
  :class:`~gEconpy.solvers.sparse_root.direction.KrylovDirection` with ``recompute_every=None`` and a
  ``preconditioner`` of your own.
- :func:`~gEconpy.solvers.sparse_root.line_search.newton_nonmonotone`: a Newton direction with nonmonotone
  backtracking, which accepts occasional increases in the residual and escapes flat regions.

Trust-region solvers, :class:`~gEconpy.solvers.sparse_root.dogleg.SparseDogleg` and
:class:`~gEconpy.solvers.sparse_root.gauss_newton.GaussNewtonTrustRegion`, adapt a trust-region radius, and
:class:`~gEconpy.solvers.sparse_root.levenberg_marquardt.LevenbergMarquardt` adapts a damping parameter. None of
the three takes a globalization strategy.

The stacked Jacobian usually factorizes fastest in its natural block-tridiagonal column order, and
:attr:`~gEconpy.model.perfect_foresight.compile.PerfectForesightProblem.column_order` names a fill-reducing order
for the models where it does not. A solver that leaves the ordering unset, which is ``linear_solver`` on a
:class:`~gEconpy.solvers.sparse_root.direction.NewtonDirection` or a
:class:`~gEconpy.solvers.sparse_root.dogleg.SparseDogleg` and ``permc_spec`` on a
:class:`~gEconpy.solvers.sparse_root.direction.ChordDirection`, solves with a copy that uses that order. The solver
passed in is not changed, and a solver that sets the ordering keeps it.

Every solver is an object with ``init`` and ``step`` methods, so a new one is a class that implements those two.
Swapping a component takes one keyword argument. This test replaces the direct sparse factorization inside a
Newton direction with GMRES:

.. literalinclude:: ../../../tests/solvers/sparse_root/test_line_search.py
   :language: python
   :pyobject: TestNewtonArmijoSpecific.test_custom_linear_solver
   :dedent: 4

Nonlinear simulation
--------------------

:func:`~gEconpy.model.perfect_foresight.extended_path.simulate_extended_path` simulates sample paths of the
nonlinear model by the extended path method. Each period draws that period's shocks, solves a perfect foresight
problem over ``horizon`` periods from the current state with every later shock set to zero, and keeps the first
period of the solution as the next state. Shocks are drawn exactly as :func:`~gEconpy.model.simulate.simulate`
draws them, so with the same arguments and seed the two simulators see the same shocks. Their paths then differ
because of the model's nonlinearity and the finite horizon described below.

.. code-block:: python

    from gEconpy import model_from_gcn, simulate_extended_path
    from gEconpy.data import get_example_gcn

    model = model_from_gcn(get_example_gcn("RBC"), verbose=False)
    paths, diagnostics = simulate_extended_path(
        model, n_simulations=10, simulation_length=100, shock_std=0.01, random_seed=0
    )

``paths`` holds levels with dimensions ``(simulation, time, variable)``. ``diagnostics`` holds, for every period,
whether it solved and the solver's work counts summed over every solve the period took.

By default every period is solved by a :func:`~gEconpy.solvers.sparse_root.line_search.chord` solver that keeps
its factorization of the stacked Jacobian from one period to the next. It refactorizes only when the residual norm
falls by less than a factor of 10 in one step, so a period often needs no factorization at all. The paths then match
those of a full Newton solver to within the solver tolerance. A different solver passed as ``solver`` changes the
trade-off, for example refactorizing less often or at every step:

.. code-block:: python

    from gEconpy.solvers.sparse_root.direction import ChordDirection
    from gEconpy.solvers.sparse_root.line_search import chord, newton_armijo

    lazy_chord = chord(
        direction=ChordDirection(recompute_every=None, refresh_ratio=0.5, keep_across_solves=True)
    )
    paths, diagnostics = simulate_extended_path(model, simulation_length=100, shock_std=0.01, solver=lazy_chord)
    paths, diagnostics = simulate_extended_path(model, simulation_length=100, shock_std=0.01, solver=newton_armijo())

Agents in the extended path are surprised by every shock and expect no further ones. The paths show how the
nonlinear model responds to the shocks that arrive, and they leave out any behavior that comes from expecting
future uncertainty. A model whose main nonlinearity is a precautionary motive, such as buffer-stock saving, shows
little of it in these paths.

Each period's problem imposes the steady state at its end, which is an approximation unless the economy has
returned to it by then. The error is first order in the shocks and shrinks with the horizon at the rate set by
the model's most persistent stable dynamics, the largest modulus among the eigenvalues of the policy matrix ``T``
from :meth:`~gEconpy.model.model.Model.solve_model`. In the RBC example above that modulus is 0.97. Against a
reference simulated with ``horizon=800``, its paths differ by up to 0.8 percent of their range at ``horizon=100``
and by less than 0.001 percent at the default of 200. The practical check is to simulate again with ``horizon``
doubled and compare. A model whose largest modulus is closer to 1 needs a longer horizon, and each period's cost
grows roughly linearly with it.

A period whose solve fails is retried by homotopy: the period's shocks and the state's deviation from the
steady state are scaled by a weight that rises from 0.1 to 1, each solve starting from the last that succeeded,
for at most ``max_homotopy_solves`` solves. If that also fails, the simulation warns, naming the simulation and
the period, and that path is NaN from the failed period on. The other paths continue. ``sparse_root_kwargs``
is passed to the solver in every period, and its ``f_tol`` defaults to :math:`10^{-5}`.

Paths are independent of each other, so ``cores`` above 1 simulates them in that many worker processes, each
taking a contiguous block of paths. Every path solves with its own copy of the solver, so the output is the same
for any number of cores. ``mp_ctx`` picks the start method, with the defaults :func:`pymc.sample` uses. A forked
worker inherits the compiled model and starts at once. Under ``"spawn"`` or ``"forkserver"`` each worker first
imports gEconpy and rebuilds the problem, which takes a few seconds, and a script that starts them needs the
usual ``if __name__ == "__main__":`` guard.

.. code-block:: python

    paths, diagnostics = simulate_extended_path(
        model, n_simulations=64, simulation_length=200, shock_std=0.01, random_seed=0, cores=8
    )

Reading the cost of a solve
---------------------------

The returned :class:`~scipy.optimize.OptimizeResult` counts the work it took. ``nit`` is accepted iterations and
``nfev`` residual evaluations, as scipy's own optimizers report them. Three more are specific to this package:
``njev`` counts Jacobian builds, ``nreject`` counts trial steps a trust-region solver threw away, and ``nsolve``
counts linear solves.

``nsolve`` is the one to watch on a large stacked system, because a sparse factorization dominates everything
else in a step. Comparing it against ``nit`` says which solver is doing what: a line-search solver reports one
solve per step, a trust-region solver reports one per trial and so one per rejection as well, and the dogleg
reports one per step no matter how many trials that step took.

The ``merit_fun`` argument of both backtracking strategies has the largest effect on perfect foresight
performance. A line search evaluates the residual at every trial step, and the full function also builds the
Jacobian. A merit function that returns the residual alone lets the search skip the Jacobian until a step is
accepted.
