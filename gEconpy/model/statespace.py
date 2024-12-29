import logging

import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
import pytensor
import pytensor.tensor as pt
import sympy as sp

from pymc.pytensorf import rewrite_pregrad
from pymc_experimental.statespace.core.statespace import PyMCStateSpace
from pymc_experimental.statespace.models.utilities import make_default_coords
from pymc_experimental.statespace.utils.constants import (
    JITTER_DEFAULT,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
)
from pytensor import graph_replace
from scipy.stats._continuous_distns import (
    beta_gen,
    expon_gen,
    gamma_gen,
    halfnorm_gen,
    invgamma_gen,
    norm_gen,
    truncnorm_gen,
    uniform_gen,
)
from scipy.stats.distributions import rv_frozen

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.perturbation import check_bk_condition_pt
from gEconpy.solvers.cycle_reduction import cycle_reduction_pt, scan_cycle_reduction
from gEconpy.solvers.gensys import gensys_pt

_log = logging.getLogger(__name__)
floatX = pytensor.config.floatX


SCIPY_TO_PRELIZ = {
    norm_gen: pz.Normal,
    halfnorm_gen: pz.HalfNormal,
    truncnorm_gen: pz.TruncatedNormal,
    uniform_gen: pz.Uniform,
    beta_gen: pz.Beta,
    gamma_gen: pz.Gamma,
    invgamma_gen: pz.InverseGamma,
    expon_gen: pz.Exponential,
}


class DSGEStateSpace(PyMCStateSpace):
    def __init__(
        self,
        variables: list[TimeAwareSymbol],
        shocks: list[TimeAwareSymbol],
        equations: list[sp.Expr],
        param_dict: dict[str, float],
        priors: list[dict[str, rv_frozen]],
        parameter_mapping: dict[pt.TensorVariable, pt.TensorVariable],
        steady_state_mapping: dict[pt.TensorVariable, pt.TensorVariable],
        ss_jac: pt.TensorVariable,
        ss_resid: pt.TensorVariable,
        ss_error: pt.TensorVariable,
        ss_error_grad: pt.TensorVariable,
        ss_error_hess: pt.TensorVariable,
        linearized_system: list[pt.TensorVariable],
    ):
        self.variables = variables
        self.equations = equations
        self.shocks = shocks
        self.priors = priors
        self.param_dict = param_dict

        self.parameter_mapping = parameter_mapping
        self.steady_state_mapping = steady_state_mapping
        self.input_parameters = [
            x for x in parameter_mapping.keys() if x.name in param_dict
        ]

        self.ss_jac = ss_jac
        self.ss_resid = ss_resid
        self.ss_error = ss_error
        self.ss_error_grad = ss_error_grad
        self.ss_error_hess = ss_error_hess

        self.linearized_system = linearized_system

        self.full_covariance = False
        self.constant_parameters = []
        self._configured = False
        self._obs_state_names = None
        self.error_states = []
        self._solver = "gensys"
        self._solver_kwargs: dict | None = None
        self._mode = None
        self._linearized_system_subbed: list | None = None
        self._policy_graph: list | None = None
        self._ss_resid: pt.TensorVariable | None = None

        self._bk_output = None
        self._policy_resid = None

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
        constant_replacements = {}
        for parameter in self.input_parameters:
            if parameter.name in self.constant_parameters:
                constant_replacements[parameter] = pt.constant(
                    np.array(self.param_dict[parameter.name]).astype(floatX),
                    name=parameter.name,
                )
            else:
                self._name_to_variable[parameter.name] = parameter

        self._linearized_system_subbed = [A, B, C, D] = pytensor.graph_replace(
            self.linearized_system, constant_replacements, strict=False
        )

        self._bk_output = check_bk_condition_pt(A, B, C, D)
        n_steps = None

        if self._solver == "gensys":
            T, R, success = gensys_pt(A, B, C, D, **self._solver_kwargs)
        elif self._solver == "cycle_reduction":
            T, R = cycle_reduction_pt(A, B, C, D, **self._solver_kwargs)
        else:
            T, R, n_steps = scan_cycle_reduction(
                A, B, C, D, mode=self._mode, **self._solver_kwargs
            )

        resid = pt.square(A + B @ T + C @ T @ T).sum()

        ss_resid = pytensor.graph_replace(
            self.ss_resid, constant_replacements, strict=False
        )
        ss_resid = pt.square(ss_resid).sum()

        T = rewrite_pregrad(T)
        R = rewrite_pregrad(R)
        resid = rewrite_pregrad(resid)
        ss_resid = rewrite_pregrad(ss_resid)

        self._policy_graph = [T, R]
        self._n_steps = n_steps
        self._policy_resid = resid
        self._ss_resid = ss_resid

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
        constant_params: list[str] | None = None,
        full_shock_covaraince: bool = False,
        solver: str = "gensys",
        mode: str | None = None,
        **solver_kwargs,
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

        # Validate constant params
        if constant_params is None:
            constant_params = []
        else:
            input_param_names = [x.name for x in self.input_parameters]
            unknown_params = [x for x in constant_params if x not in input_param_names]
            if len(unknown_params) > 0:
                raise ValueError(
                    f'The following parameters are unknown to the model and cannot be set as constant: '
                    f'{", ".join(unknown_params)}'
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
        self.constant_parameters = constant_params

        self.full_covariance = full_shock_covaraince
        self._configured = True
        self._solver = solver
        self._solver_kwargs = solver_kwargs
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
        if self.constant_parameters is not None:
            param_names = [x for x in param_names if x not in self.constant_parameters]

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

        for var in self.param_names:
            placeholder = self._name_to_variable[var]

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

    def build_statespace_graph(
        self,
        data: np.ndarray | pd.DataFrame | pt.TensorVariable,
        register_data: bool = True,
        missing_fill_value: float | None = None,
        cov_jitter: float | None = JITTER_DEFAULT,
        save_kalman_filter_outputs_in_idata: bool = False,
        add_norm_check: bool = True,
        add_bk_check: bool = False,
        add_solver_success_check: bool = False,
        add_steady_state_penalty: bool = True,
        resid_penalty: float = 1.0,
    ) -> None:
        super().build_statespace_graph(
            data=data,
            register_data=register_data,
            missing_fill_value=missing_fill_value,
            cov_jitter=cov_jitter,
            save_kalman_filter_outputs_in_idata=save_kalman_filter_outputs_in_idata,
            mode=self._mode,
        )

        pymc_model = pm.modelcontext(None)

        replacement_dict = {
            var: pymc_model[name] for name, var in self._name_to_variable.items()
        }

        A, B, C, D, T, R = graph_replace(
            self._linearized_system_subbed + self._policy_graph,
            replace=replacement_dict,
            strict=False,
        )

        if self._n_steps is not None:
            n_steps = graph_replace(
                self._n_steps, replace=replacement_dict, strict=False
            )
            pm.Deterministic("n_cycle_steps", n_steps.astype(int))

        policy_resid, *bk_output, ss_resid = graph_replace(
            [self._policy_resid, *self._bk_output, self._ss_resid],
            replace=replacement_dict,
            strict=False,
        )

        bk_flag, n_forward, n_gt_one = bk_output

        if add_norm_check:
            n_vars, n_shocks = R.shape
            tm1_grid = np.array(
                [
                    [eq.has(var.set_t(-1)) for var in self.variables]
                    for eq in self.equations
                ]
            )
            t_grid = np.array(
                [
                    [eq.has(var.set_t(0)) for var in self.variables]
                    for eq in self.equations
                ]
            )

            tm1_idx = np.any(tm1_grid, axis=0)
            t_idx = np.any(t_grid, axis=0)

            shock_idx = pt.arange(n_shocks)
            state_var_mask = pt.bitwise_and(tm1_idx, t_idx)

            QQ = R[:n_vars, :]
            P = T[state_var_mask, :][:, state_var_mask]
            Q = QQ[state_var_mask, :][:, shock_idx]

            A_prime = A[:, state_var_mask]
            R_prime = T[:, state_var_mask]
            S_prime = QQ[:, shock_idx]

            norm_deterministic = pm.Deterministic(
                "deterministic_norm",
                pt.linalg.norm(A_prime + B @ R_prime + C @ R_prime @ P),
            )
            norm_stochastic = pm.Deterministic(
                "stochastic_norm", pt.linalg.norm(B @ S_prime + C @ R_prime @ Q + D)
            )

            # Add penalty terms to the likelihood to rule out invalid solutions
            pm.Potential(
                "solution_norm_penalty",
                -resid_penalty * (norm_deterministic + norm_stochastic),
            )

        if add_bk_check:
            pm.Deterministic("bk_flag", bk_flag)
            pm.Potential(
                "bk_condition_satisfied", pt.switch(pt.eq(bk_flag, 1.0), 0.0, -np.inf)
            )

        if add_solver_success_check:
            policy_resid = pm.Deterministic("policy_resid", policy_resid)
            pm.Potential("policy_resid_penalty", -resid_penalty * policy_resid)

        if add_steady_state_penalty:
            ss_resid = pm.Deterministic("ss_resid", ss_resid)
            pm.Potential("steady_state_resid_penalty", -resid_penalty * ss_resid)

    def priors_to_preliz(self):
        priors = self.priors[0]
        pz_priors = {}

        for name, rv in priors.items():
            dist_type = type(rv.dist)
            pz_dist = SCIPY_TO_PRELIZ[dist_type]

            match rv.dist:
                case norm_gen():
                    pz_priors[name] = pz_dist(mu=rv.kwds["loc"], sigma=rv.kwds["scale"])
                case truncnorm_gen():
                    loc, scale, a, b = (rv.kwds[x] for x in ["loc", "scale", "a", "b"])
                    lower = loc + scale * a
                    upper = loc + scale * b
                    pz_priors[name] = pz_dist(
                        mu=loc, sigma=scale, lower=lower, upper=upper
                    )
                case halfnorm_gen():
                    pz_priors[name] = pz_dist(sigma=rv.kwds["scale"])
                case gamma_gen():
                    pz_priors[name] = pz_dist(
                        alpha=rv.kwds["a"], beta=1 / rv.kwds["scale"]
                    )
                case beta_gen():
                    pz_priors[name] = pz_dist(alpha=rv.kwds["a"], beta=rv.kwds["b"])
                case uniform_gen():
                    pz_priors[name] = pz_dist(lower=rv.kwds["a"], upper=rv.kwds["b"])
                case invgamma_gen():
                    pz_priors[name] = pz_dist(alpha=rv.kwds["a"], beta=rv.kwds["scale"])
                case expon_gen():
                    pz_priors[name] = pz_dist(lam=1 / rv.kwds["scale"])

        return pz_priors
