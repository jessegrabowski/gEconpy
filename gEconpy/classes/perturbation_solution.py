from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from gEconpy.model.perturbation import DROrder


@dataclass(frozen=True, eq=False)
class FirstOrderPerturbationSolution:
    """
    First-order perturbation solution of a DSGE model.

    The policy function is stored as Dynare stores it, with rows in decision-rule order and only the state columns
    of the transition matrix. :attr:`T` and :attr:`R` rebuild the full matrices in model order. Iterating the
    solution yields ``T`` and ``R``, so ``T, R = model.solve_model()`` keeps working.

    Attributes
    ----------
    ghx : ndarray
        Response of every variable to the lagged state variables, of shape ``(n_variables, n_state_variables)``.
        Rows follow ``dr_order.var_order`` and columns follow the state variables in the same order.
    ghu : ndarray
        Response of every variable to the current shocks, of shape ``(n_variables, n_shocks)``, with rows in
        ``dr_order.var_order``.
    steady_state : ndarray
        Steady-state value of every variable, in model order.
    log_linearized : ndarray of bool
        Whether each variable, in model order, was log-linearized, so that its deviations are in logs.
    variable_names : list of str
        Variable names in model order.
    shock_names : list of str
        Shock names in model order.
    dr_order : DROrder
        Decision-rule ordering of the model.
    """

    ghx: np.ndarray
    ghu: np.ndarray
    steady_state: np.ndarray
    log_linearized: np.ndarray
    variable_names: list[str]
    shock_names: list[str]
    dr_order: DROrder

    @classmethod
    def from_policy_matrices(
        cls,
        T: np.ndarray,
        R: np.ndarray,
        steady_state: np.ndarray,
        log_linearized: np.ndarray,
        variable_names: list[str],
        shock_names: list[str],
        dr_order: DROrder,
    ) -> "FirstOrderPerturbationSolution":
        """
        Build the solution from full transition and selection matrices in model order.

        Parameters
        ----------
        T : ndarray
            Transition matrix, of shape ``(n_variables, n_variables)``.
        R : ndarray
            Selection matrix, of shape ``(n_variables, n_shocks)``.
        steady_state : ndarray
            Steady-state value of every variable, in model order.
        log_linearized : ndarray of bool
            Whether each variable, in model order, was log-linearized.
        variable_names : list of str
            Variable names in model order.
        shock_names : list of str
            Shock names in model order.
        dr_order : DROrder
            Decision-rule ordering of the model.

        Returns
        -------
        solution : FirstOrderPerturbationSolution
            The solution, keeping only the state columns of ``T``.
        """
        var_order = dr_order.var_order
        state_columns = var_order[dr_order.lag_positions]

        return cls(
            ghx=np.ascontiguousarray(T[np.ix_(var_order, state_columns)]),
            ghu=np.ascontiguousarray(R[var_order]),
            steady_state=steady_state,
            log_linearized=log_linearized,
            variable_names=variable_names,
            shock_names=shock_names,
            dr_order=dr_order,
        )

    @property
    def T(self) -> np.ndarray:
        """Transition matrix in model order, zero in the columns of variables that never appear at a lag."""
        var_order = self.dr_order.var_order
        state_columns = var_order[self.dr_order.lag_positions]

        n_variables = len(self.variable_names)
        T = np.zeros((n_variables, n_variables), dtype=self.ghx.dtype)
        T[np.ix_(var_order, state_columns)] = self.ghx
        return T

    @property
    def R(self) -> np.ndarray:
        """Selection matrix in model order."""
        R = np.empty_like(self.ghu)
        R[self.dr_order.var_order] = self.ghu
        return R

    def to_levels(self, deviations: np.ndarray) -> np.ndarray:
        """
        Convert deviations from the steady state into levels.

        A log-linearized variable's deviation is a log deviation, so its level is ``y_ss * exp(deviation)``. Every
        other variable's level is ``y_ss + deviation``.

        Parameters
        ----------
        deviations : ndarray
            Deviations from the steady state, with variables in model order on the last axis.

        Returns
        -------
        levels : ndarray
            Levels of the variables, with the shape of ``deviations``.
        """
        deviations = np.asarray(deviations, dtype=float)
        levels = self.steady_state + deviations
        logged = self.log_linearized
        levels[..., logged] = self.steady_state[logged] * np.exp(deviations[..., logged])
        return levels

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.T
        yield self.R
