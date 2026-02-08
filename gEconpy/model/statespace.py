import logging
import warnings

from typing import Literal, get_args

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import sympy as sp
import xarray as xr

from preliz.distributions.distributions import Distribution
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.pytensorf import rewrite_pregrad
from pymc_extras.statespace.core.properties import Coord, Parameter, Shock, State, SymbolicVariable
from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.utils.constants import (
    JITTER_DEFAULT,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
)
from pytensor import graph_replace

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.distributions import CompositeDistribution
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.perturbation import check_bk_condition_pt
from gEconpy.solvers.cycle_reduction import cycle_reduction_pt, scan_cycle_reduction
from gEconpy.solvers.gensys import gensys_pt

_log = logging.getLogger(__name__)
floatX = pytensor.config.floatX

SolverType = Literal["gensys", "cycle_reduction", "scan_cycle_reduction"]
valid_solvers = get_args(SolverType)


class DSGEStateSpace(PyMCStateSpace):
    """Core class for estimating DSGE models using PyMC."""

    def __init__(
        self,
        variables: list[TimeAwareSymbol],
        shocks: list[TimeAwareSymbol],
        equations: list[sp.Expr],
        param_dict: dict[str, float],
        hyper_param_dict: dict[str, float],
        param_priors: SymbolDictionary[str, Distribution],
        shock_priors: SymbolDictionary[str, CompositeDistribution],
        parameter_mapping: dict[pt.TensorVariable, pt.TensorVariable],
        steady_state_mapping: dict[pt.TensorVariable, pt.TensorVariable],
        ss_jac: pt.TensorVariable,
        ss_resid: pt.TensorVariable,
        ss_error: pt.TensorVariable,
        ss_error_grad: pt.TensorVariable,
        ss_error_hess: pt.TensorVariable,
        linearized_system: list[pt.TensorVariable],
        verbose: bool = True,
    ):
        """
        Create a :class:`pmx.statespace.PyMCStateSpace` model representing a linearized DSGE.

        Users should not create this class direction, and should instead use
        :func:`gEconpy.model.build.statespace_from_gcn` to compile a statespace model from a gcn file.

        Parameters
        ----------
        variables: list of TimeAwareSymbol
            List of variables in the model
        shocks: list of TimeAwareSymbol
            List of shocks in the model
        equations: list of sympy.Expr
            List of equations in the model
        param_dict: dict
            Dictionary of default parameter values, as defined in the model file
        hyper_param_dict: dict
            Dictionary of default hyperparameter values, as defined in the model file
        param_priors: dict
            Dictionary of preliz parameter priors
        shock_priors: dict
            Dictionary of preliz shock priors
        parameter_mapping: dict
            Symbolic function mapping input parameters to the full vector of parameters, including
            deterministic.
        steady_state_mapping: dict
            Symbolic function mapping input parameters to the steady state values of the model
        ss_jac: pt.TensorVariable
            Symbolic Jacobian of the steady state equations
        ss_resid: pt.TensorVariable
            Symbolic vector of (signed) residuals of the steady state equations
        ss_error: pt.TensorVariable
            Symbolic scalar-valued error function used to minimize the steady state residuals.
        ss_error_grad: pt.TensorVariable
            Symbolic gradient of the steady state error function
        ss_error_hess: pt.TensorVariable
            Symbolic hessian of the steady state error function
        linearized_system: list of pt.TensorVariable
            List of four symbolic expressions representing the linearized system of equations as partial
            jacobians of the model equations with respect to variables at time t+1 (A), t (B), t-1 (C), and with
            respect to exogenous shocks (D), each evaluated at the (symbolic) steady state.
        verbose: bool
            If True, show diagnostic messages.
        """
        self.variables = variables
        self.equations = equations
        self.shocks = shocks
        self.param_priors = param_priors
        self.shock_priors = shock_priors
        self.param_dict = param_dict
        self.hyper_param_dict = hyper_param_dict

        self.parameter_mapping = parameter_mapping
        self.steady_state_mapping = steady_state_mapping
        self.input_parameters = [x for x in parameter_mapping if x.name in param_dict]

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
        self._n_steps = None

        self.verbose = verbose

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

        for variable in self.input_parameters:
            self._tensor_variable_info = self._tensor_variable_info.add(
                SymbolicVariable(name=variable.name, symbolic_variable=variable)
            )

    def _setup_policy_matrices(
        self, A: pt.TensorVariable, B: pt.TensorVariable, C: pt.TensorVariable, D: pt.TensorVariable
    ) -> tuple[pt.TensorVariable, pt.TensorVariable]:
        if self._solver == "gensys":
            T, R, _success = gensys_pt(A, B, C, D, **self._solver_kwargs)
        elif self._solver == "cycle_reduction":
            T, R = cycle_reduction_pt(A, B, C, D, **self._solver_kwargs)
        else:
            T, R, n_steps = scan_cycle_reduction(A, B, C, D, mode=self._mode, **self._solver_kwargs)
            self._n_steps = n_steps

        return T, R

    def _setup_state_covariance(self):
        if self.full_covariance:
            state_cov = self.make_and_register_variable("state_cov", shape=(self.k_posdef, self.k_posdef))
            self.ssm["state_cov", :, :] = state_cov
            return

        for i, shock in enumerate(self.shocks):
            sigma = self.make_and_register_variable(f"sigma_{shock.base_name}", shape=())
            self.ssm["state_cov", i, i] = sigma**2

    def _make_design_matrix(self):
        Z = np.zeros((self.k_endog, self.k_states))

        for i, name in enumerate(self.observed_states):
            Z[i, self.state_names.index(name)] = 1.0

        return Z

    def make_symbolic_graph(self):
        if not self._configured:
            if self.verbose:
                _log.info("Statespace model construction complete, but call the .configure method to finalize.")
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

        T, R = self._setup_policy_matrices(A, B, C, D)
        resid = pt.square(A + B @ T + C @ T @ T).sum()

        ss_resid = pytensor.graph_replace(self.ss_resid, constant_replacements, strict=False)
        ss_resid = pt.square(ss_resid).sum()

        T = rewrite_pregrad(T)
        R = rewrite_pregrad(R)
        resid = rewrite_pregrad(resid)
        ss_resid = rewrite_pregrad(ss_resid)

        self._policy_graph = [T, R]
        self._policy_resid = resid
        self._ss_resid = ss_resid

        self.ssm["transition", :, :] = T
        self.ssm["selection", :, :] = R
        self.ssm["design", :, :] = self._make_design_matrix()

        self._setup_state_covariance()

        if self.measurement_error:
            for i, state in enumerate(self.error_states):
                sigma = self.make_and_register_variable(f"error_sigma_{state}", shape=())
                self.ssm["obs_cov", i, i] = sigma**2

        self.ssm["initial_state", :] = pt.zeros(self.k_states)

        Q = self.ssm["state_cov"]
        method = "direct" if self.use_direct_lyapunov else "bilinear"
        self.ssm["initial_state_cov", :, :] = pt.linalg.solve_discrete_lyapunov(T, R @ Q @ R.T, method=method)

    def configure(
        self,
        observed_states: list[str],
        measurement_error: list[str] | None = None,
        constant_params: list[str] | None = None,
        full_shock_covaraince: bool = False,
        solver: SolverType = "gensys",
        mode: str | None = None,
        verbose=True,
        max_iter: int = 50,
        tol: float = 1e-6,
        use_adjoint_gradients: bool = True,
        use_direct_lyapunov: bool = False,
    ):
        # Set up observed states
        unknown_states = [x for x in observed_states if x not in self.state_names]
        if len(unknown_states) > 0:
            raise ValueError(
                f"The following states are unknown to the model and cannot be set as observed: "
                f"{', '.join(unknown_states)}"
            )

        # Set up measurement errors
        if measurement_error is None:
            measurement_error = []
        else:
            unknown_states = [x for x in measurement_error if x not in observed_states]
            if len(unknown_states) > 0:
                raise ValueError(
                    f"The following states are not observed, and cannot have measurement error: "
                    f"{', '.join(unknown_states)}"
                )

        # Validate constant params
        if constant_params is None:
            constant_params = []
        else:
            input_param_names = [x.name for x in self.input_parameters]
            unknown_params = [x for x in constant_params if x not in input_param_names]
            if len(unknown_params) > 0:
                raise ValueError(
                    f"The following parameters are unknown to the model and cannot be set as constant: "
                    f"{', '.join(unknown_params)}"
                )

        # Validate solver argument
        if solver not in valid_solvers:
            raise ValueError(
                f'Unknown solver {solver}, expected one of "gensys", "cycle_reduction", or "scan_cycle_reduction"'
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

        if solver == "gensys":
            solver_kwargs = {"tol": tol}
        elif solver == "cycle_reduction":
            solver_kwargs = {"tol": tol, "max_iter": max_iter}
        else:
            solver_kwargs = {
                "tol": tol,
                "max_iter": max_iter,
                "use_adjoint_gradients": use_adjoint_gradients,
            }

        self._obs_state_names = observed_states
        self.error_states = measurement_error
        self.constant_parameters = constant_params

        self.full_covariance = full_shock_covaraince
        self.use_direct_lyapunov = use_direct_lyapunov
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
            verbose=verbose,
        )

        for variable in self.input_parameters:
            self._tensor_variable_info = self._tensor_variable_info.add(
                SymbolicVariable(name=variable.name, symbolic_variable=variable)
            )

    def set_states(self) -> tuple[State, ...]:
        observed_states = self._obs_state_names if self._obs_state_names is not None else []
        hidden_states = [State(name=x.base_name, observed=False) for x in self.variables]
        observed_states = [State(name=name, observed=True) for name in observed_states]
        return *hidden_states, *observed_states

    def set_parameters(self) -> tuple[Parameter, ...]:
        # TODO: Extract information from assumptions and use them to denote constraints on the parameters
        constant_params = self.constant_parameters if self.constant_parameters is not None else []
        parameters = [Parameter(name=x.name, shape=()) for x in self.input_parameters if x.name not in constant_params]

        if self.full_covariance:
            parameters += [
                Parameter(
                    name="state_cov",
                    shape=(self.k_posdef, self.k_posdef),
                    dims=(SHOCK_DIM, SHOCK_AUX_DIM),
                    constraints="Positive Semi-Definite",
                ),
            ]
        else:
            parameters += [
                Parameter(name=f"sigma_{shock.base_name}", shape=(), constraints="Positive") for shock in self.shocks
            ]

        if self.measurement_error:
            parameters += [
                Parameter(name=f"error_sigma_{state}", shape=(), constraints="Positive") for state in self.error_states
            ]

        return tuple(parameters)

    def set_shocks(self) -> tuple[Shock]:
        return tuple(Shock(name=x.base_name) for x in self.shocks)

    def set_coords(self) -> tuple[Coord, ...]:
        return self.default_coords()

    @property
    def param_dims(self):
        if not self._configured:
            return {}

        return {param: None if param != "state_cov" else (SHOCK_DIM, SHOCK_AUX_DIM) for param in self.param_names}

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

        replacement_dict = {var: pymc_model[name] for name, var in self._name_to_variable.items()}

        A, B, C, D, T, R = graph_replace(
            self._linearized_system_subbed + self._policy_graph,
            replace=replacement_dict,
            strict=False,
        )

        if self._n_steps is not None:
            n_steps = graph_replace(self._n_steps, replace=replacement_dict, strict=False)
            pm.Deterministic("n_cycle_steps", n_steps.astype(int))

        policy_resid, *bk_output, ss_resid = graph_replace(
            [self._policy_resid, *self._bk_output, self._ss_resid],
            replace=replacement_dict,
            strict=False,
        )

        bk_flag, _n_forward, _n_gt_one = bk_output

        if add_norm_check:
            n_vars, n_shocks = R.shape
            tm1_grid = np.array([[eq.has(var.set_t(-1)) for var in self.variables] for eq in self.equations])
            t_grid = np.array([[eq.has(var.set_t(0)) for var in self.variables] for eq in self.equations])

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
            norm_stochastic = pm.Deterministic("stochastic_norm", pt.linalg.norm(B @ S_prime + C @ R_prime @ Q + D))

            # Add penalty terms to the likelihood to rule out invalid solutions
            pm.Potential(
                "solution_norm_penalty",
                -resid_penalty * (norm_deterministic + norm_stochastic),
            )

        if add_bk_check:
            pm.Deterministic("bk_flag", bk_flag)
            pm.Potential("bk_condition_satisfied", pt.switch(pt.eq(bk_flag, 1.0), 0.0, -np.inf))

        if add_solver_success_check:
            policy_resid = pm.Deterministic("policy_resid", policy_resid)
            pm.Potential("policy_resid_penalty", -resid_penalty * policy_resid)

        if add_steady_state_penalty:
            ss_resid = pm.Deterministic("ss_resid", ss_resid)
            pm.Potential("steady_state_resid_penalty", -resid_penalty * ss_resid)

    def to_pymc(self, exclude_priors: list[str] | None = None):
        if exclude_priors is None:
            exclude_priors = []

        with pm.modelcontext(None):
            for prior, dist in self.param_priors.items():
                if prior in exclude_priors:
                    continue
                dist.to_pymc(name=prior)

            for prior, dist in self.shock_priors.items():
                if prior in exclude_priors:
                    continue
                dist.to_pymc()


def data_from_prior(
    statepace_mod: DSGEStateSpace,
    pymc_model: pm.Model,
    index: pd.DatetimeIndex | None = None,
    n_samples: int = 500,
    pct_missing: float = 0,
    random_seed: np.random.Generator | int | None = None,
) -> tuple[xr.Dataset, pd.DataFrame, az.InferenceData]:
    """
    Generate artificial data from prior predictive samples.

    Also modifies the pymc model and the statespace model in-place to act as if build_statespace_graph has been
    called with the new data.

    Parameters
    ----------
    statepace_mod: DSGEStateSpace
        Statespace model to generate data from. Must have been configured with the .configure method.
    pymc_model: pm.Model
        PyMC model with priors on expected DSGE parameters. It should **not** have a Kalman Filter added via
        build_statespace_graph.
    index: pd.DatetimeIndex
        Index to use for the generated data. If None, a quarterly index from 1980-01-01 to 2024-11-01 is used.
    n_samples: int
        Number of prior predictive samples to draw.
    pct_missing: float
        Percentage of missing data to introduce into the generated data. Must be between 0 and 1.
    random_seed: np.random.Generator or int, optional
        Random number generator to use for sampling. If None, the default numpy random number generator is used.

    Returns
    -------
    true_parameters: xr.Dataset
        True parameters used to generate the data.
    data: pd.DataFrame
        Generated data.
    prior_idata: az.InferenceData
        Draws from the prior predictive distribution, plus conditional prior predictive samples.
    """
    rng = np.random.default_rng(random_seed)

    if index is None:
        index = pd.date_range(start="1980-01-01", end="2024-11-01", freq="QS-OCT")
    dummy_data = pd.DataFrame(np.nan, index=index, columns=statepace_mod.observed_states)
    dummy_data.index.freq = dummy_data.index.inferred_freq

    # Copy the model so the original model is unchanged
    new_model = pymc_model.copy()

    with new_model:
        if "data" not in new_model:
            statepace_mod.build_statespace_graph(
                dummy_data,
                add_bk_check=False,
                add_solver_success_check=True,
                add_norm_check=True,
                add_steady_state_penalty=True,
            )
        else:
            pm.set_data({"data": dummy_data.fillna(-9999)})

    with warnings.catch_warnings(action="ignore"), freeze_dims_and_data(new_model):
        prior_idata = pm.sample_prior_predictive(n_samples, compile_kwargs={"mode": "JAX"}, random_seed=rng)

    with warnings.catch_warnings(action="ignore"):
        prior_trajectories = statepace_mod.sample_unconditional_prior(prior_idata, random_seed=rng)

    prior_idata["unconditional_prior"] = prior_trajectories

    idx = rng.choice(prior_idata.prior.coords["draw"].values)

    true_params = prior_idata.prior.isel(chain=0, draw=idx)
    true_params["param_idx"] = idx

    data = prior_trajectories.isel(chain=0, draw=idx).prior_observed
    data = data.to_dataframe().drop(columns=["chain", "draw"]).unstack("observed_state").droplevel(axis=1, level=0)

    data.index.freq = data.index.inferred_freq
    if pct_missing > 0:
        n_missing = int(data.shape[0] * pct_missing)
        for col in data:
            missing_idxs = rng.choice(data.index, size=n_missing, replace=False)
            data.loc[missing_idxs, col] = np.nan

    # Reset the statespace model so the user can call build_statespace_graph with the new data
    statepace_mod._fit_data = None
    statepace_mod._fit_dims = None
    statepace_mod._fit_coords = None

    return true_params, data, prior_idata
