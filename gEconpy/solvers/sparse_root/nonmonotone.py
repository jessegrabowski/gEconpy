from dataclasses import dataclass, field

import numpy as np

from gEconpy.solvers.sparse_root.base import IterationStats, RootFunction, SolverState, StepInfo, merit
from gEconpy.solvers.sparse_root.direction import NewtonDirection
from gEconpy.solvers.sparse_root.globalization import NonmonotoneBacktracking


@dataclass
class NewtonNonmonotone:
    """Newton solver with Grippo-Lampariello-Lucidi nonmonotone line search.

    Identical to ``NewtonArmijo`` except the Armijo condition compares against
    the maximum merit over a sliding window of past iterates, allowing
    occasional increases in the objective. This can help escape narrow valleys
    where monotone line search stalls.

    Parameters
    ----------
    direction : NewtonDirection
        Direction computation strategy.
    globalization : NonmonotoneBacktracking
        Nonmonotone backtracking line search.
    """

    direction: NewtonDirection = field(default_factory=NewtonDirection)
    globalization: NonmonotoneBacktracking = field(default_factory=NonmonotoneBacktracking)

    def init(self, fun: RootFunction, x0: np.ndarray, args: tuple) -> SolverState:
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
