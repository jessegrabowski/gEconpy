import logging

import numpy as np
import pytensor.tensor as pt
import sympy as sp

from pymc_experimental.statespace.core.statespace import PyMCStateSpace
from pymc_experimental.statespace.models.utilities import make_default_coords
from pymc_experimental.statespace.utils.constants import SHOCK_AUX_DIM, SHOCK_DIM

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.solvers.cycle_reduction import cycle_reduction_pt, scan_cycle_reduction
from gEconpy.solvers.gensys import gensys_pt

_log = logging.getLogger(__name__)


class DSGEStateSpace(PyMCStateSpace):
    def __init__(
        self,
        variables: list[TimeAwareSymbol],
        shocks: list[TimeAwareSymbol],
        equations: list[sp.Expr],
        param_dict: dict[str, float],
        input_parameters: list[pt.TensorVariable],
        deterministic_params: list[pt.TensorVariable],
        calibrated_params: list[pt.TensorVariable],
        steady_state_solutions: pt.TensorVariable,
        ss_jac: pt.TensorVariable,
        ss_resid: pt.TensorVariable,
        ss_error: pt.TensorVariable,
        ss_error_grad: pt.TensorVariable,
        ss_error_hess: pt.TensorVariable,
        linearized_system: list[pt.TensorVariable],
    ):
        # Store some info
        self.variables = variables
        self.equations = equations
        self.shocks = shocks
        self.param_dict = param_dict

        self.input_parameters = input_parameters
        self.deterministic_parameters = deterministic_params
        self.caibrated_parameters = calibrated_params

        self.steady_state_solutions = steady_state_solutions
        self.ss_jac = ss_jac
        self.ss_resid = ss_resid
        self.ss_error = ss_error
        self.ss_error_grad = ss_error_grad
        self.ss_error_hess = ss_error_hess

        self.linearized_system = linearized_system

        self.full_covariance = False
        self._configured = False
        self._obs_state_names = None
        self.error_states = []
        self._solver = "gensys"
        self._mode = None

        if len(calibrated_params) > 0:
            raise NotImplementedError(
                "Calibration not yet implemented in StateSpace model"
            )

        # Check that the entire steady state has been provided
        n_ss_eqs = steady_state_solutions.type.shape[0]
        if n_ss_eqs != len(variables):
            raise NotImplementedError(
                "Numeric steady state not yet implemented in StateSpace model"
            )

        k_endog = 1  # to be updated later
        k_states = len(variables)
        k_posdef = len(shocks)

        super().__init__(
            k_endog,
            k_states,
            k_posdef,
            filter_type="standard",
            verbose=False,
            measurement_error=False,
        )

    def make_symbolic_graph(self):
        if not self._configured:
            _log.info(
                "Statespace model construction complete, but call the .configure method to finalize."
            )
            return

        # Register the existing placeholders with the statespace model
        for parameter in self.input_parameters:
            self._name_to_variable[parameter.name] = parameter

        A, B, C, D = self.linearized_system

        if self._solver == "gensys":
            T, R, success = gensys_pt(A, B, C, D)
        elif self._solver == "cycle_reduction":
            T, R, resid = cycle_reduction_pt(A, B, C, D)
        else:
            T, R, resid = scan_cycle_reduction(A, B, C, D, mode=self._mode)

        self.ssm["transition", :, :] = T
        self.ssm["selection", :, :] = R
        self.ssm["design", :, :] = self._make_design_matrix()

        if not self.full_covariance:
            for i, shock in enumerate(self.shocks):
                sigma = self.make_and_register_variable(
                    f"sigma_{shock.base_name}", shape=()
                )
                self.ssm["state_cov", i, i] = sigma**2
        else:
            state_cov = self.make_and_register_variable(
                "state_cov", shape=(self.k_posdef, self.k_posdef)
            )
            self.ssm["state_cov", :, :] = state_cov

        if self.measurement_error:
            for i, state in enumerate(self.error_states):
                sigma = self.make_and_register_variable(f"sigma_{state}", shape=())
                self.ssm["obs_cov", i, i] = sigma**2

        self.ssm["initial_state", :] = pt.zeros(self.k_states)

        Q = self.ssm["state_cov"]
        self.ssm["initial_state_cov", :, :] = pt.linalg.solve_discrete_lyapunov(
            T, R @ Q @ R.T
        )

    def configure(
        self,
        observed_states: list[str],
        measurement_error: list[str] | None = None,
        full_shock_covaraince: bool = False,
        solver: str = "gensys",
        mode: str | None = None,
    ):
        # Set up observed states
        unknown_states = [x for x in observed_states if x not in self.state_names]
        if len(unknown_states) > 0:
            raise ValueError(
                f'The following states are unknown to the model and cannot be set as observed: '
                f'{", ".join(unknown_states)}'
            )

        # Set up measurement errors
        if measurement_error is None:
            measurement_error = []
        else:
            unknown_states = [x for x in measurement_error if x not in observed_states]
            if len(unknown_states) > 0:
                raise ValueError(
                    f'The following states are not observed, and cannot have measurement error: '
                    f'{", ".join(unknown_states)}'
                )

        # Validate solver argument
        if solver not in ["gensys", "cycle_reduction", "scan_cycle_reduction"]:
            raise ValueError(
                f'Unknown solver {solver}, expected one of "gensys", "cycle_reduction", '
                f'or "scan_cycle_reduction"'
            )

        # Check model is identified
        k_endog = len(observed_states)
        model_df = len(measurement_error) + len(self.shock_names)
        verb = "are" if model_df != 1 else "is"
        suffix = "s" if model_df != 1 else ""
        if k_endog > model_df:
            raise ValueError(
                f"Stochastic singularity! You requested {k_endog} observed timeseries, but there {verb} "
                f"only {model_df} source{suffix} of stochastic variation. "
                f"\n\nReduce the number of observed timeseries, or add more sources of stochastic "
                f"variation (by adding measurement error or structural shocks)"
            )

        self._obs_state_names = observed_states
        self.error_states = measurement_error

        self.full_covariance = full_shock_covaraince
        self._configured = True
        self._solver = solver
        self._mode = mode

        # Rebuild the internal statespace representation and kalman filters with the newly resized matrices
        super().__init__(
            k_endog,
            self.k_states,
            self.k_posdef,
            measurement_error=len(measurement_error) > 0,
            verbose=True,
        )

    def _make_design_matrix(self):
        Z = np.zeros((self.k_endog, self.k_states))

        for i, name in enumerate(self.observed_states):
            Z[i, self.state_names.index(name)] = 1.0

        return Z

    @property
    def param_names(self):
        param_names = [x.name for x in self.input_parameters]
        if self.full_covariance:
            param_names += ["state_cov"]
        else:
            param_names += [f"sigma_{shock.base_name}" for shock in self.shocks]

        if self.measurement_error:
            param_names += [f"sigma_{state}" for state in self.error_states]

        return param_names

    @property
    def state_names(self):
        return [x.base_name for x in self.variables]

    @property
    def shock_names(self):
        return [x.base_name for x in self.shocks]

    @property
    def observed_states(self):
        return self._obs_state_names

    @property
    def param_dims(self):
        if not self._configured:
            return {}

        return {
            param: None if param != "state_cov" else (SHOCK_DIM, SHOCK_AUX_DIM)
            for param in self.param_names
        }

    @property
    def coords(self):
        coords = make_default_coords(self)
        return coords

    @property
    def param_info(self):
        info = {}
        if not self._configured:
            return info

        for var, placeholder in self._name_to_variable.items():
            info[var] = {
                "shape": placeholder.type.shape,
                "initval": self.param_dict.get(var, None),
            }
            if var.startswith("sigma"):
                info[var]["constraints"] = "Positive"
            elif var == "state_cov":
                info[var]["constraints"] = "Positive Semi-Definite"
            else:
                info[var]["constraints"] = None

        # Lazy way to add the dims without making any typos
        for name in self.param_names:
            info[name]["dims"] = self.param_dims[name]

        return info
