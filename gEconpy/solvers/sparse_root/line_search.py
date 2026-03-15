"""Unified line-search solver for sparse root finding.

This module provides the core LineSearchSolver that composes a direction
strategy with a globalization strategy. The named solvers (NewtonArmijo,
Chord, etc.) are simple aliases with different defaults.
"""

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
    """Generic line-search solver composing direction + globalization strategies.

    This is the core solver implementation. The named solvers (NewtonArmijo,
    Chord, InexactNewtonKrylov, NewtonNonmonotone) are aliases with different
    default strategies.

    Parameters
    ----------
    direction : DirectionStrategy
        Strategy for computing the search direction. Must implement
        ``compute(x, res, jac) -> DirectionProposal``.
    globalization : GlobalizationStrategy
        Line search strategy. Must implement
        ``search(fun, x, phi, proposal, args) -> LineSearchResult``.
    """

    direction: DirectionStrategy = field(default_factory=NewtonDirection)
    globalization: GlobalizationStrategy = field(default_factory=ArmijoBacktracking)

    def init(self, fun: RootFunction, x0: np.ndarray, args: tuple) -> SolverState:
        # Reset stateful direction strategies
        if hasattr(self.direction, "reset"):
            self.direction.reset()

        x = np.asarray(x0, dtype=np.float64).copy()
        res, jac = fun(x, *args)
        return SolverState(x=x, res=res, jac=jac, phi=merit(res), stats=IterationStats(nfev=1, njev=1))

    def step(self, fun: RootFunction, state: SolverState, args: tuple) -> tuple[SolverState, StepInfo]:
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
    """Newton's method with Armijo backtracking line search."""
    return LineSearchSolver(
        direction=direction if direction is not None else NewtonDirection(),
        globalization=globalization if globalization is not None else ArmijoBacktracking(),
    )


def Chord(
    direction: DirectionStrategy | None = None,
    globalization: GlobalizationStrategy | None = None,
) -> LineSearchSolver:
    """Chord method (cached Jacobian) with Armijo line search."""
    return LineSearchSolver(
        direction=direction if direction is not None else ChordDirection(),
        globalization=globalization if globalization is not None else ArmijoBacktracking(),
    )


def InexactNewtonKrylov(
    direction: DirectionStrategy | None = None,
    globalization: GlobalizationStrategy | None = None,
) -> LineSearchSolver:
    """Inexact Newton with Krylov direction (GMRES/BiCGSTAB)."""
    return LineSearchSolver(
        direction=direction if direction is not None else KrylovDirection(),
        globalization=globalization if globalization is not None else ArmijoBacktracking(),
    )


def NewtonNonmonotone(
    direction: DirectionStrategy | None = None,
    globalization: GlobalizationStrategy | None = None,
) -> LineSearchSolver:
    """Newton with nonmonotone (Grippo-Lampariello-Lucidi) line search."""
    return LineSearchSolver(
        direction=direction if direction is not None else NewtonDirection(),
        globalization=globalization if globalization is not None else NonmonotoneBacktracking(),
    )
