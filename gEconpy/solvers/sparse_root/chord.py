from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from gEconpy.solvers.sparse_root.base import (
    DEFAULT_ARMIJO_BETA,
    DEFAULT_ARMIJO_C1,
    DEFAULT_ARMIJO_MAX_ITER,
    IterationStats,
    RootFunction,
    SolverState,
    StepInfo,
    merit,
)
from gEconpy.solvers.sparse_root.direction import ChordDirection
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking


@dataclass
class Chord:
    """Chord method: Newton direction with a cached Jacobian.

    Reuses the Jacobian from a previous iteration for ``recompute_every`` steps,
    reducing the cost of direction computation when Jacobian factorization is
    expensive. The residual and Armijo condition are evaluated with up-to-date
    information every step.

    Parameters
    ----------
    direction : ChordDirection
        Direction strategy with Jacobian caching.
    globalization : ArmijoBacktracking
        Line search globalization.
    """

    direction: ChordDirection = field(default_factory=ChordDirection)
    globalization: ArmijoBacktracking = field(default_factory=ArmijoBacktracking)

    def init(self, fun: RootFunction, x0: np.ndarray, args: tuple) -> SolverState:
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


def chord(
    recompute_every: int = 5,
    linear_solver: Callable | None = None,
    c1: float = DEFAULT_ARMIJO_C1,
    beta: float = DEFAULT_ARMIJO_BETA,
    max_iter: int = DEFAULT_ARMIJO_MAX_ITER,
) -> Chord:
    """Create a Chord solver with common configuration."""
    return Chord(
        direction=ChordDirection(linear_solver=linear_solver, recompute_every=recompute_every),
        globalization=ArmijoBacktracking(c1=c1, beta=beta, max_iter=max_iter),
    )
