from dataclasses import dataclass, field

import numpy as np

from gEconpy.solvers.sparse_root.base import IterationStats, RootFunction, SolverState, StepInfo, merit
from gEconpy.solvers.sparse_root.direction import KrylovDirection
from gEconpy.solvers.sparse_root.globalization import ArmijoBacktracking


@dataclass
class InexactNewtonKrylov:
    """Newton solver with inexact Krylov direction and Armijo line search.

    Uses GMRES or BiCGSTAB to solve the Newton system approximately.
    The Krylov tolerance is adapted via the Eisenstat-Walker formula
    so early iterations are cheap (loose solve) and final iterations
    are accurate (tight solve).

    Parameters
    ----------
    direction : KrylovDirection
        Krylov direction computation strategy.
    globalization : ArmijoBacktracking
        Line search globalization.
    """

    direction: KrylovDirection = field(default_factory=KrylovDirection)
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
