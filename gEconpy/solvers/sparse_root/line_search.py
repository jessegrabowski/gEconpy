from dataclasses import dataclass, field

import numpy as np

from gEconpy.solvers.sparse_root.base import (
    IterationStats,
    RootFunction,
    SolverState,
    StepInfo,
    merit,
)
from gEconpy.solvers.sparse_root.direction import (
    ChordDirection,
    DirectionStrategy,
    KrylovDirection,
    NewtonDirection,
)
from gEconpy.solvers.sparse_root.globalization import (
    ArmijoBacktracking,
    GlobalizationStrategy,
    NonmonotoneBacktracking,
)


@dataclass
class LineSearchSolver:
    """Generic line-search solver composing a direction strategy with a globalization strategy.

    The named solvers :func:`NewtonArmijo`, :func:`Chord`, :func:`InexactNewtonKrylov` and
    :func:`NewtonNonmonotone` build instances of this class with different default strategies.

    Parameters
    ----------
    direction : DirectionStrategy
        Strategy for computing the search direction. Must implement ``compute(x, res, jac)``, returning a
        direction proposal. Defaults to :class:`~gEconpy.solvers.sparse_root.direction.NewtonDirection`.
    globalization : GlobalizationStrategy
        Line search strategy. Must implement ``search(fun, x, phi, proposal, args)``, returning a line search
        result. Defaults to :class:`~gEconpy.solvers.sparse_root.globalization.ArmijoBacktracking`.
    """

    direction: DirectionStrategy = field(default_factory=NewtonDirection)
    globalization: GlobalizationStrategy = field(default_factory=ArmijoBacktracking)

    def init(self, fun: RootFunction, x0: np.ndarray, args: tuple) -> SolverState:
        """Evaluate ``fun`` at ``x0`` and build the initial solver state.

        Parameters
        ----------
        fun : callable
            Fused residual and Jacobian function, called as ``fun(x, *args)``.
        x0 : ndarray
            Initial guess for the root.
        args : tuple
            Extra positional arguments passed to ``fun``.

        Returns
        -------
        state : ~gEconpy.solvers.sparse_root.base.SolverState
            State holding the initial point, residuals, Jacobian, merit value and evaluation counts.
        """
        # Reset stateful direction strategies
        if hasattr(self.direction, "reset"):
            self.direction.reset()

        x = np.asarray(x0, dtype=np.float64).copy()
        res, jac = fun(x, *args)
        return SolverState(x=x, res=res, jac=jac, phi=merit(res), stats=IterationStats(nfev=1, njev=1))

    def step(self, fun: RootFunction, state: SolverState, args: tuple) -> tuple[SolverState, StepInfo]:
        """Take one line-search iteration from ``state``.

        Parameters
        ----------
        fun : callable
            Fused residual and Jacobian function, called as ``fun(x, *args)``.
        state : ~gEconpy.solvers.sparse_root.base.SolverState
            Current solver state.
        args : tuple
            Extra positional arguments passed to ``fun``.

        Returns
        -------
        new_state : ~gEconpy.solvers.sparse_root.base.SolverState
            State after the accepted step, or the unchanged input state when the line search fails.
        info : ~gEconpy.solvers.sparse_root.base.StepInfo
            Whether the step was accepted, the step taken, and a failure message when it was not.
        """
        proposal = self.direction.compute(state.x, state.res, state.jac)

        try:
            ls = self.globalization.search(fun, state.x, state.phi, proposal, args)
        except RuntimeError as e:
            return state, StepInfo(accepted=False, step=np.zeros_like(state.x), message=str(e))

        step = ls.alpha * proposal.direction
        new_state = SolverState(
            x=ls.x_new,
            res=ls.res_new,
            jac=ls.jac_new,
            phi=ls.phi_new,
            stats=state.stats.update(nit=1, nfev=ls.n_evals, njev=ls.n_evals, nsolve=1),
        )
        return new_state, StepInfo(accepted=True, step=step)


def NewtonArmijo(
    direction: DirectionStrategy | None = None,
    globalization: GlobalizationStrategy | None = None,
) -> LineSearchSolver:
    """Build a Newton solver with Armijo backtracking line search.

    Parameters
    ----------
    direction : DirectionStrategy, optional
        Search direction strategy. Defaults to
        :class:`~gEconpy.solvers.sparse_root.direction.NewtonDirection`.
    globalization : GlobalizationStrategy, optional
        Line search strategy. Defaults to
        :class:`~gEconpy.solvers.sparse_root.globalization.ArmijoBacktracking`.

    Returns
    -------
    solver : LineSearchSolver
        Solver composing the two strategies.
    """
    return LineSearchSolver(
        direction=direction if direction is not None else NewtonDirection(),
        globalization=globalization if globalization is not None else ArmijoBacktracking(),
    )


def Chord(
    direction: DirectionStrategy | None = None,
    globalization: GlobalizationStrategy | None = None,
) -> LineSearchSolver:
    """Build a Chord solver, which reuses a cached Jacobian, with Armijo backtracking line search.

    Parameters
    ----------
    direction : DirectionStrategy, optional
        Search direction strategy. Defaults to
        :class:`~gEconpy.solvers.sparse_root.direction.ChordDirection`.
    globalization : GlobalizationStrategy, optional
        Line search strategy. Defaults to
        :class:`~gEconpy.solvers.sparse_root.globalization.ArmijoBacktracking`.

    Returns
    -------
    solver : LineSearchSolver
        Solver composing the two strategies.
    """
    return LineSearchSolver(
        direction=direction if direction is not None else ChordDirection(),
        globalization=globalization if globalization is not None else ArmijoBacktracking(),
    )


def InexactNewtonKrylov(
    direction: DirectionStrategy | None = None,
    globalization: GlobalizationStrategy | None = None,
) -> LineSearchSolver:
    """Build an inexact Newton solver that computes the direction with a Krylov method.

    Parameters
    ----------
    direction : DirectionStrategy, optional
        Search direction strategy. Defaults to
        :class:`~gEconpy.solvers.sparse_root.direction.KrylovDirection`, which uses GMRES.
    globalization : GlobalizationStrategy, optional
        Line search strategy. Defaults to
        :class:`~gEconpy.solvers.sparse_root.globalization.ArmijoBacktracking`.

    Returns
    -------
    solver : LineSearchSolver
        Solver composing the two strategies.
    """
    return LineSearchSolver(
        direction=direction if direction is not None else KrylovDirection(),
        globalization=globalization if globalization is not None else ArmijoBacktracking(),
    )


def NewtonNonmonotone(
    direction: DirectionStrategy | None = None,
    globalization: GlobalizationStrategy | None = None,
) -> LineSearchSolver:
    """Build a Newton solver with a nonmonotone (Grippo-Lampariello-Lucidi) line search.

    Parameters
    ----------
    direction : DirectionStrategy, optional
        Search direction strategy. Defaults to
        :class:`~gEconpy.solvers.sparse_root.direction.NewtonDirection`.
    globalization : GlobalizationStrategy, optional
        Line search strategy. Defaults to
        :class:`~gEconpy.solvers.sparse_root.globalization.NonmonotoneBacktracking`.

    Returns
    -------
    solver : LineSearchSolver
        Solver composing the two strategies.
    """
    return LineSearchSolver(
        direction=direction if direction is not None else NewtonDirection(),
        globalization=globalization if globalization is not None else NonmonotoneBacktracking(),
    )
