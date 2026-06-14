import logging
import warnings

from typing import Literal

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
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    JITTER_DEFAULT,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
)
from pytensor.assumptions import assume
from pytensor.graph.replace import graph_replace
from sympytensor import as_tensor

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.distributions import CompositeDistribution
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.perturbation import check_bk_condition_pt
from gEconpy.parser.grammar.expressions import parse_expression
from gEconpy.parser.transform.to_sympy import ast_to_sympy
from gEconpy.pytensorf.block import block
from gEconpy.solvers.backward_looking import solve_policy_function_with_backward_direct_pt
from gEconpy.solvers.cycle_reduction import cycle_reduction_pt, scan_cycle_reduction
from gEconpy.solvers.gensys import gensys_pt

_log = logging.getLogger(__name__)
floatX = pytensor.config.floatX

VALID_SOLVERS = ("gensys", "cycle_reduction", "scan_cycle_reduction", "backward_direct")
VALID_AGGREGATIONS = ("sum", "mean", "first", "last")
CUMULATOR_AGGREGATIONS = ("sum", "mean")


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
        linearized_system: list[pt.TensorVariable],
        var_order: np.ndarray | None = None,
        log_linearized_variables: list[str] | None = None,
        sympytensor_cache: dict | None = None,
        filter_type: str = "standard",
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
        linearized_system: list of pt.TensorVariable
            List of four symbolic expressions representing the linearized system of equations as partial
            jacobians of the model equations with respect to variables at time t+1 (A), t (B), t-1 (C), and with
            respect to exogenous shocks (D), each evaluated at the (symbolic) steady state.
        log_linearized_variables: list of str, optional
            Base names of variables that were log-linearized when building ``linearized_system``. Used by
            ``configure(ss_obs_intercept=...)`` to decide whether an observation intercept entry is
            ``log(v_ss(p))`` (log-linearized) or ``v_ss(p)`` (level-linearized).
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

        self.linearized_system = linearized_system

        # Variable column permutation applied by ``linearize_model`` to expose A's and C's
        # block-zero column structure. T/R returned by the solver have rows AND columns
        # in this permuted order; the solver call site applies ``inv_var_order`` to put
        # them back into the user's variable order before they reach the Kalman path.
        if var_order is None:
            var_order = np.arange(len(variables))
        self.var_order = np.asarray(var_order, dtype=int)
        self.inv_var_order = np.argsort(self.var_order)

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

        self._bk_output = None
        self._policy_resid = None
        self._n_steps = None
        self._lead_var_idx: np.ndarray | None = None

        # Mixed-frequency state augmentation
        self._temporal_aggregation: dict[str, str] = {}
        self._aggregation_period: int = 4
        self._k_orig_states: int = len(variables)

        self._log_linearized_variables: set[str] = set(log_linearized_variables or [])
        self._ss_obs_intercept_states: list[str] = []

        # Shared sympytensor cache from ``statespace_from_gcn``. Reusing it lets
        # subsequent sympy-to-pytensor conversions (observation equations)
        # reference the same TensorVariables that key ``steady_state_mapping``
        # and ``parameter_mapping``.
        self._sympytensor_cache: dict = sympytensor_cache if sympytensor_cache is not None else {}

        # Linearized obs equations: maps obs-series name to
        # ``(intercept_pt, {(var_base_name, lag): coeff_pt})``. Pytensor
        # expressions in the input-parameter placeholders. Populated by
        # ``configure`` when ``observation_equations`` is supplied.
        self._obs_equations: dict[str, tuple] = {}

        # Per-variable lag depth required by any observation equation.
        # Computed at ``configure`` time alongside the linearization.
        self._obs_lag_depths: dict[str, int] = {}

        # Per-variable starting column of its obs-eq lag chain in the
        # augmented state vector. Populated at ``configure`` time.
        self._obs_lag_starts: dict[str, int] = {}

        self.verbose = verbose

        k_endog = 1  # to be updated later
        k_states = len(variables)
        k_posdef = len(shocks)

        super().__init__(
            k_endog,
            k_states,
            k_posdef,
            filter_type=filter_type,
            verbose=False,
            measurement_error=False,
        )

        for variable in self.input_parameters:
            self._tensor_variable_info = self._tensor_variable_info.add(
                SymbolicVariable(name=variable.name, symbolic_variable=variable)
            )

    def _setup_policy_matrices(
        self, A: pt.TensorVariable, B: pt.TensorVariable, C: pt.TensorVariable, D: pt.TensorVariable
    ) -> tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
        if self._solver == "gensys":
            T, R, _success = gensys_pt(A, B, C, D, **self._solver_kwargs)
        elif self._solver == "cycle_reduction":
            T, R = cycle_reduction_pt(A, B, C, D, **self._solver_kwargs)
        elif self._solver == "backward_direct":
            T, R = solve_policy_function_with_backward_direct_pt(A, B, C, D)
        else:
            T, R, n_steps = scan_cycle_reduction(A, B, C, D, mode=self._mode, **self._solver_kwargs)
            self._n_steps = n_steps

        # Evaluate the policy-function residual in the solver's variable order. A, B, and C
        # have columns in ``var_order``, and T shares that basis until it is remapped below;
        # computing the residual after the remap would mix the two orderings and inflate it.
        resid = pt.square(A + B @ T + C @ T @ T).sum()

        # T comes back in the *permuted* variable order on both axes; R on its rows.
        # Map back to the user's variable order so the Kalman filter sees user variables.
        if not np.array_equal(self.var_order, np.arange(len(self.var_order))):
            inv = self.inv_var_order
            T = T[inv][:, inv]
            R = R[inv]

        return T, R, resid

    @property
    def lead_var_idx(self) -> np.ndarray:
        """Column indices of forward-looking variables (variables appearing at t+1 in any equation)."""
        if self._lead_var_idx is None:
            idx = []
            for i, v in enumerate(self.variables):
                if any(eq.has(v.set_t(1)) for eq in self.equations):
                    idx.append(i)
            self._lead_var_idx = np.array(idx, dtype=int)
        return self._lead_var_idx

    @property
    def n_forward(self) -> int:
        """Number of forward-looking variables."""
        return len(self.lead_var_idx)

    def _setup_state_covariance(self):
        """Build the ``state_cov`` SSM matrix and return it.

        The returned matrix is also the one the Lyapunov solve for ``initial_state_cov``
        consumes.
        """
        if self.full_covariance:
            state_cov = self.make_and_register_variable("state_cov", shape=(self.k_posdef, self.k_posdef))
            Q = assume(state_cov, positive_definite=True)
            self.ssm["state_cov"] = Q
            return Q

        # ``pt.diag(stack(...))`` is auto-tagged diagonal by AssumptionFeature, which propagates
        # symmetric/PSD through the congruence rule (R Q R'), enabling cholesky-based solves and
        # significant speedups in compile_dlogp for HMC sampling.
        sigmas = [self.make_and_register_variable(f"sigma_{shock.base_name}", shape=()) for shock in self.shocks]
        Q = pt.diag(pt.stack([s**2 for s in sigmas]))
        self.ssm["state_cov"] = Q
        return Q

    def _make_design_matrix(self):
        """
        Build the observation design matrix :math:`Z`.

        For each observed state, fill the row with the linearized coefficients
        from a user-supplied observation equation (parameter-dependent), or with
        a selector entry (unit weight, or :math:`1/s` for ``mean`` aggregation)
        on the state's column plus its cumulator slots if any. Returns the
        constant numpy form when no observation equations are configured and a
        pytensor matrix otherwise.

        Returns
        -------
        Z : ndarray or pt.TensorVariable
            Constant ``(k_endog, k_states)`` array when every observed state
            uses the selector form, or a pytensor matrix of the same shape when
            any observation equation contributes parameter-dependent
            coefficients.
        """
        n_cum_lags = self._aggregation_period - 1
        cumulator_vars = self._cumulator_variables

        if not self._obs_equations:
            # Pure selector design — keep the existing constant-numpy path.
            Z = np.zeros((self.k_endog, self.k_states))
            for i, name in enumerate(self.observed_states):
                orig_idx = self._orig_state_names.index(name)
                agg_method = self._temporal_aggregation.get(name)
                if agg_method in CUMULATOR_AGGREGATIONS:
                    agg_pos = cumulator_vars.index(name)
                    cum_start = self._k_orig_states + agg_pos * n_cum_lags
                    weight = 1.0 / self._aggregation_period if agg_method == "mean" else 1.0
                    Z[i, orig_idx] = weight
                    Z[i, cum_start : cum_start + n_cum_lags] = weight
                else:
                    Z[i, orig_idx] = 1.0
            return Z

        # At least one observation equation in play — build Z symbolically.
        # Use ``inc_subtensor`` for the obs-equation rows so that overlapping
        # contributions from ``(lag, d)`` aggregation pairs accumulate
        # (e.g. annual-summed quarterly log-differences telescope).
        Z_sym = pt.zeros((self.k_endog, self.k_states), dtype=floatX)
        for i, name in enumerate(self.observed_states):
            agg_method = self._temporal_aggregation.get(name)

            if name in self._obs_equations:
                _, coeffs = self._obs_equations[name]
                if agg_method in CUMULATOR_AGGREGATIONS:
                    n_periods = self._aggregation_period
                    coeff_weight = 1.0 if agg_method == "sum" else 1.0 / n_periods
                else:
                    n_periods = 1
                    coeff_weight = 1.0
                for (vname, lag), coeff_pt in coeffs.items():
                    for d in range(n_periods):
                        effective_lag = lag - d
                        col = (
                            self._orig_state_names.index(vname)
                            if effective_lag == 0
                            else self._obs_lag_column(vname, effective_lag)
                        )
                        Z_sym = pt.inc_subtensor(Z_sym[i, col], coeff_weight * coeff_pt)
            else:
                weight = 1.0 / self._aggregation_period if agg_method == "mean" else 1.0
                orig_idx = self._orig_state_names.index(name)
                Z_sym = pt.set_subtensor(Z_sym[i, orig_idx], weight)
                if agg_method in CUMULATOR_AGGREGATIONS:
                    agg_pos = cumulator_vars.index(name)
                    cum_start = self._k_orig_states + agg_pos * n_cum_lags
                    for k in range(n_cum_lags):
                        Z_sym = pt.set_subtensor(Z_sym[i, cum_start + k], weight)
        return Z_sym

    def _make_obs_intercept(self) -> pt.TensorVariable:
        r"""
        Build the observation-intercept vector :math:`d`.

        For each observed state :math:`v`:

        - If :math:`v` has a user-supplied observation equation, the entry is
          the linearization's constant term — for example
          :math:`\log Y_{ss}(p) + \log Z_{ss}(p)` for the BGP growth-rate
          observation.
        - Else if :math:`v` is in ``self._ss_obs_intercept_states``, the entry
          is :math:`\log v_{ss}(p)` (log-linearized) or :math:`v_{ss}(p)`
          (level-linearized).
        - Otherwise the entry is zero, appropriate when the data for that
          series is already in deviation form (HP-cycled, demeaned, etc.).

        Temporal aggregation: ``sum``-aggregated observations get the
        per-period intercept multiplied by ``aggregation_period``; ``mean``,
        ``first``, ``last``, and the default no-aggregation case keep the
        single-period value.

        Returns
        -------
        d : pt.TensorVariable
            Length-``k_endog`` vector of intercepts in the model's input-
            parameter graph.
        """
        ss_by_name = {k.name: v for k, v in self.steady_state_mapping.items()}
        ss_set = set(self._ss_obs_intercept_states)
        entries: list[pt.TensorVariable] = []
        for name in self.observed_states:
            if name in self._obs_equations:
                intercept_pt, _ = self._obs_equations[name]
                base = intercept_pt
            elif name in ss_set:
                ss_key = f"{name}_ss"
                if ss_key not in ss_by_name:
                    raise ValueError(
                        f"ss_obs_intercept requested for {name!r}, but no symbolic steady state "
                        f"is available for it. This usually means the variable was eliminated "
                        f"by tryreduce or has no analytic SS."
                    )
                v_ss_expr = ss_by_name[ss_key]
                base = pt.log(v_ss_expr) if name in self._log_linearized_variables else v_ss_expr
            else:
                entries.append(pt.zeros((), dtype=floatX))
                continue

            agg = self._temporal_aggregation.get(name)
            if agg == "sum":
                entries.append(self._aggregation_period * base)
            else:
                entries.append(base)

        return pt.stack(entries).astype(floatX)

    def _parse_observation_equation(self, name: str, expr_str: str) -> sp.Expr:
        """
        Parse a GCN-syntax observation equation into a sympy expression in the model's namespace.

        Accepts contemporaneous and lagged variable references (``v[]``,
        ``v[-1]``, ...). Leads raise ``ValueError`` — an observation cannot
        depend on the future.

        Parameters
        ----------
        name : str
            Observed-series name the equation belongs to, used for error
            messages.
        expr_str : str
            GCN-syntax expression in terms of model variables and parameters.

        Returns
        -------
        sym : sympy.Expr
            Parsed expression with free symbols resolved against the model's
            variable and parameter namespaces.
        """
        ast = parse_expression(expr_str, context=f"observation_equations[{name!r}]")
        # Carry the model variables' assumptions (positive, etc.) through to the
        # parsed symbols. Sympy equality and hashing include assumptions, so
        # without this the parsed TimeAwareSymbols would not compare equal to
        # the model's, and the linearization's ``xreplace`` would silently
        # leave them in place.
        assumptions = {v.base_name: dict(v.assumptions0) for v in self.variables}
        sym = ast_to_sympy(ast, assumptions=assumptions)

        var_names = {v.base_name for v in self.variables}
        param_names = set(self.param_dict) | set(self.hyper_param_dict)

        for s in sym.free_symbols:
            if isinstance(s, TimeAwareSymbol):
                if s.time_index == "ss":
                    continue
                if s.time_index > 0:
                    raise ValueError(
                        f"Observation equation {name!r} contains a lead reference "
                        f"{s}. Only contemporaneous and lagged model variables are "
                        f"allowed."
                    )
                if s.base_name not in var_names:
                    raise ValueError(
                        f"Observation equation {name!r} references unknown model "
                        f"variable {s.base_name!r}. Known: {sorted(var_names)}"
                    )
            elif s.name not in param_names:
                raise ValueError(
                    f"Observation equation {name!r} references unknown symbol "
                    f"{s.name!r}: not a model variable, parameter, or hyperparameter."
                )
        return sym

    def _linearize_observation_equation(self, sym: sp.Expr) -> tuple[sp.Expr, dict[tuple[str, int], sp.Expr]]:
        r"""
        First-order linearize an observation equation around the model's steady state.

        Each variable reference :math:`v_{t+k}` in ``sym`` (with :math:`k \le 0`)
        is substituted with :math:`v_{ss} \exp(\tilde v_k)` (log-linearized
        variables) or :math:`v_{ss} + \tilde v_k` (level-linearized variables),
        where :math:`\tilde v_k` is a fresh dummy. The intercept is the value
        at all :math:`\tilde v_k = 0`; the coefficient on each :math:`\tilde
        v_k` is the first partial derivative there. The coefficient
        corresponds to the contribution of :math:`v`'s deviation at lag
        :math:`k` in the (augmented) state vector, so for a log-linearized
        :math:`v` it equals
        :math:`v_{ss}\, \partial g / \partial v_{t+k}\big|_{ss}` (chain rule).

        Parameters
        ----------
        sym : sympy.Expr
            Observation equation in raw (un-linearized) form, in the model's
            symbol namespace.

        Returns
        -------
        intercept : sympy.Expr
            Constant term :math:`g(x_{ss}, p)` in steady-state symbols and
            parameters.
        coeffs : dict mapping (str, int) to sympy.Expr
            Maps each appearing ``(variable_base_name, time_index)`` pair to
            its linear coefficient. ``time_index`` is ``0`` for contemporaneous
            references and negative for lags.
        """
        var_by_name = {v.base_name: v for v in self.variables}
        appearing = {
            (s.base_name, s.time_index)
            for s in sym.free_symbols
            if isinstance(s, TimeAwareSymbol) and s.time_index != "ss"
        }

        forward: dict[TimeAwareSymbol, sp.Expr] = {}
        tildes: dict[tuple[str, int], sp.Symbol] = {}
        for vname, lag in appearing:
            v = var_by_name[vname]
            v_at_t = v.set_t(lag)
            v_ss = v.set_t("ss")
            lag_tag = "0" if lag == 0 else f"m{-lag}"
            v_tilde = sp.Symbol(f"_tilde_{vname}_{lag_tag}", real=True)
            tildes[(vname, lag)] = v_tilde
            if vname in self._log_linearized_variables:
                forward[v_at_t] = v_ss * sp.exp(v_tilde)
            else:
                forward[v_at_t] = v_ss + v_tilde

        g_sub = sym.xreplace(forward)
        zero_subs = {tilde: sp.Integer(0) for tilde in tildes.values()}

        intercept = g_sub.xreplace(zero_subs)
        coeffs: dict[tuple[str, int], sp.Expr] = {}
        for key, tilde in tildes.items():
            coeff = sp.diff(g_sub, tilde).xreplace(zero_subs)
            coeffs[key] = coeff

        return intercept, coeffs

    def _obs_eq_to_pytensor(
        self, intercept_sym: sp.Expr, coeffs_sym: dict[tuple[str, int], sp.Expr]
    ) -> tuple[pt.TensorVariable, dict[tuple[str, int], pt.TensorVariable]]:
        """
        Convert the sympy linearization output to pytensor expressions in the model's input parameters.

        Uses ``self._sympytensor_cache`` so that the resulting TensorVariables
        for steady-state symbols and parameters share identity with the
        existing keys in ``self.steady_state_mapping`` and
        ``self.parameter_mapping``. A subsequent ``graph_replace`` with
        ``steady_state_mapping`` swaps each steady-state symbol for its
        expression in input parameters.

        Parameters
        ----------
        intercept_sym : sympy.Expr
            Constant term of the linearization.
        coeffs_sym : dict mapping (str, int) to sympy.Expr
            Coefficient of each appearing ``(variable, lag)`` pair.

        Returns
        -------
        intercept_pt : pt.TensorVariable
            Pytensor scalar in input-parameter placeholders.
        coeffs_pt : dict mapping (str, int) to pt.TensorVariable
            Pytensor scalars in input-parameter placeholders, one per
            ``(variable, lag)`` pair.
        """

        def to_tensor(sym_expr):
            tv = as_tensor(sym_expr, self._sympytensor_cache)
            # sympy integers (e.g. 0 or 1) come through as Python ints — wrap.
            if not isinstance(tv, pt.Variable):
                tv = pt.as_tensor_variable(tv)
            return tv

        intercept_pt = to_tensor(intercept_sym)
        coeffs_pt = {key: to_tensor(c_sym) for key, c_sym in coeffs_sym.items()}

        # Substitute SS-tensor placeholders with their parameter-dependent
        # expressions, then cast to floatX (sympy 0/1 come through as int constants).
        ss_replace = dict(self.steady_state_mapping)
        intercept_pt = pt.cast(graph_replace(intercept_pt, ss_replace, strict=False), floatX)
        coeffs_pt = {
            vname: pt.cast(graph_replace(c, ss_replace, strict=False), floatX) for vname, c in coeffs_pt.items()
        }
        return intercept_pt, coeffs_pt

    @property
    def _n_cumulator_states(self) -> int:
        return len(self._cumulator_variables) * (self._aggregation_period - 1)

    @property
    def _cumulator_variables(self) -> list[str]:
        # Cumulator aggregation of an observation equation lives in the obs-eq
        # lag block (since the obs-series name may not match a model variable);
        # exclude those from this list so ``_augment_transition`` doesn't try
        # to index them in the model-variable namespace.
        return [
            var
            for var, method in self._temporal_aggregation.items()
            if method in CUMULATOR_AGGREGATIONS and var not in self._obs_equations
        ]

    @property
    def _orig_state_names(self) -> list[str]:
        return [x.base_name for x in self.variables]

    @property
    def _cumulator_state_names(self) -> list[str]:
        return [
            f"{var}_cumulator_lag{lag}"
            for var in self._cumulator_variables
            for lag in range(1, self._aggregation_period)
        ]

    @property
    def _n_obs_lag_states(self) -> int:
        return sum(self._obs_lag_depths.values())

    @property
    def _obs_lag_state_names(self) -> list[str]:
        return [f"{var}_obs_lag{k}" for var, depth in self._obs_lag_depths.items() for k in range(1, depth + 1)]

    def _obs_lag_column(self, var_name: str, lag: int) -> int:
        """Return the augmented-state column index for ``var_name`` lagged by ``-lag`` (lag<0)."""
        depth = -lag
        return self._obs_lag_starts[var_name] + (depth - 1)

    def _augment_transition(self, T: pt.TensorVariable) -> pt.TensorVariable:
        """
        Augment the transition matrix with cumulator rows/columns for temporally aggregated variables.

        The augmented matrix has the block form::

            T_aug = [ T  |  0              ]
                    [----|-----------------|
                    [ F  |  kron(I_n, C)   ]

        where ``C`` is the ``(s-1) × (s-1)`` lower-shift companion matrix (constant, shared
        by all aggregated variables), and ``F`` is a loading matrix with unit selectors that
        copy each lagged variable value into the first cumulator position.

        Parameters
        ----------
        T : pt.TensorVariable
            Original k_orig x k_orig transition matrix from the perturbation solution.

        Returns
        -------
        T_aug : pt.TensorVariable
            Augmented (k_orig + n_cum) x (k_orig + n_cum) transition matrix.
        """
        cumulator_vars = self._cumulator_variables
        if not cumulator_vars:
            return T

        k_orig = self._k_orig_states
        n_agg = len(cumulator_vars)
        n_cum_lags = self._aggregation_period - 1
        n_cum = self._n_cumulator_states

        shift = np.zeros((n_cum_lags, n_cum_lags), dtype=floatX)
        if n_cum_lags > 1:
            shift[np.arange(1, n_cum_lags), np.arange(n_cum_lags - 1)] = 1.0
        C = np.kron(np.eye(n_agg, dtype=floatX), shift)

        agg_indices = [self._orig_state_names.index(name) for name in cumulator_vars]
        F = pt.zeros((n_cum, k_orig), dtype=floatX)
        for agg_pos, orig_idx in enumerate(agg_indices):
            F = pt.set_subtensor(F[agg_pos * n_cum_lags, orig_idx], 1.0)

        # Build via ``block`` so ``local_block_dot_to_block_of_dots`` can split
        # downstream ``T_aug @ x`` into block-of-dots and drop the zero top-right
        # block contribution entirely.
        zero_block = pt.zeros((k_orig, n_cum), dtype=floatX)
        return block(
            [
                [T, zero_block],
                [F, pt.constant(C)],
            ]
        )

    def _append_obs_lag_block(self, T_aug: pt.TensorVariable) -> pt.TensorVariable:
        """
        Append shift-companion chains for variables referenced at non-zero lag in obs equations.

        For each variable :math:`v` with required lag depth :math:`d`, append
        :math:`d` slots; slot 1 copies :math:`v` from the previous time step,
        and each subsequent slot copies the prior slot. The first slot's row
        of the augmented transition selects the column corresponding to
        :math:`v`'s entry in the existing augmented state vector.

        Parameters
        ----------
        T_aug : pt.TensorVariable
            Augmented transition matrix, with the cumulator block already
            appended.

        Returns
        -------
        T_aug : pt.TensorVariable
            Same matrix with the obs-eq lag block appended in the trailing
            rows and columns.
        """
        n_obs_lag = self._n_obs_lag_states
        if n_obs_lag == 0:
            return T_aug

        k_prev = self._k_orig_states + self._n_cumulator_states
        F_lag = pt.zeros((n_obs_lag, k_prev), dtype=floatX)
        C_lag = pt.zeros((n_obs_lag, n_obs_lag), dtype=floatX)
        for vname, depth in self._obs_lag_depths.items():
            orig_idx = self._orig_state_names.index(vname)
            block_start = self._obs_lag_starts[vname] - k_prev
            F_lag = pt.set_subtensor(F_lag[block_start, orig_idx], 1.0)
            for k in range(1, depth):
                C_lag = pt.set_subtensor(C_lag[block_start + k, block_start + k - 1], 1.0)

        zero_block = pt.zeros((k_prev, n_obs_lag), dtype=floatX)
        return block(
            [
                [T_aug, zero_block],
                [F_lag, C_lag],
            ]
        )

    def _augment_selection(self, R: pt.TensorVariable) -> pt.TensorVariable:
        """
        Augment the selection matrix with cumulator rows for temporally aggregated variables.

        The augmented matrix has the block form::

            R_aug = [ R ]
                    [---]
                    [ 0 ]

        The cumulator rows are all zeros because cumulators are deterministic lag copies.

        Parameters
        ----------
        R : pt.TensorVariable
            Original k_orig x k_posdef selection matrix.

        Returns
        -------
        R_aug : pt.TensorVariable
            Augmented (k_orig + n_cum) x k_posdef selection matrix.
        """
        n_extra = self._n_cumulator_states + self._n_obs_lag_states
        if n_extra == 0:
            return R

        zeros = pt.zeros((n_extra, self.k_posdef), dtype=floatX)
        return pt.join(-2, R, zeros)

    def make_symbolic_graph(self):
        """
        Build the symbolic statespace graph for the DSGE model.

        This method constructs the PyTensor computational graph representing the linearized DSGE model
        in state-space form. It sets up the transition and selection matrices from the perturbation
        solution, configures the observation equation, and initializes state covariances.

        The method should only be called after :meth:`configure` has been called.
        """
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

        self._linearized_system_subbed = [A, B, C, D] = graph_replace(
            self.linearized_system, constant_replacements, strict=False
        )

        # Apply the same constant substitution to any cached obs-equation tensors so
        # ``constant_params`` parameters are baked in there too (otherwise they would
        # remain as free graph inputs and ``compile_logp`` would fail to bind them).
        if constant_replacements and self._obs_equations:
            self._obs_equations = {
                name: (
                    graph_replace(intercept_pt, constant_replacements, strict=False),
                    {v: graph_replace(c, constant_replacements, strict=False) for v, c in coeffs_pt.items()},
                )
                for name, (intercept_pt, coeffs_pt) in self._obs_equations.items()
            }

        # A/B/C have columns in ``var_order`` (D's columns are shocks). Translate
        # ``lead_var_idx`` from original variable positions to permuted positions.
        permuted_lead_var_idx = self.inv_var_order[self.lead_var_idx]
        self._bk_output = check_bk_condition_pt(A, B, C, D, lead_var_idx=permuted_lead_var_idx)

        T, R, resid = self._setup_policy_matrices(A, B, C, D)

        T = rewrite_pregrad(T)
        R = rewrite_pregrad(R)
        resid = rewrite_pregrad(resid)

        self._policy_graph = [T, R]
        self._policy_resid = resid

        T_aug = self._augment_transition(T)
        T_aug = self._append_obs_lag_block(T_aug)
        R_aug = self._augment_selection(R)

        self.ssm["transition"] = T_aug
        self.ssm["selection"] = R_aug
        self.ssm["design"] = self._make_design_matrix()
        if self._ss_obs_intercept_states or self._obs_equations:
            obs_intercept = self._make_obs_intercept()
            # ``_make_obs_intercept`` reads ``self.steady_state_mapping`` directly for the
            # ``ss_obs_intercept`` branch — those SS expressions are in free parameter
            # placeholders, so ``constant_params`` leaks unless we bake constants in
            # here too.
            if constant_replacements:
                obs_intercept = graph_replace(obs_intercept, constant_replacements, strict=False)
            self.ssm["obs_intercept"] = obs_intercept

        Q = self._setup_state_covariance()

        if self.measurement_error:
            sigmas = [self.make_and_register_variable(f"error_sigma_{state}", shape=()) for state in self.error_states]
            variances = pt.stack([s**2 for s in sigmas])
            if len(sigmas) == self.k_endog:
                H = pt.diag(variances)
            else:
                # Mirror the previous semantics: sigmas land at positions 0..len(error_states)-1
                # of an (k_endog, k_endog) zero matrix.
                diag_vec = pt.zeros((self.k_endog,))[: len(sigmas)].set(variances)
                H = pt.diag(diag_vec)
            self.ssm["obs_cov"] = H

        self.ssm["initial_state"] = pt.zeros(self.k_states)

        method = "direct" if self.use_direct_lyapunov else "bilinear"
        P0 = pt.linalg.solve_discrete_lyapunov(T_aug, R_aug @ Q @ R_aug.T, method=method)
        # The solve already propagates symmetry; deterministic cumulator/obs-lag copies make the
        # stationary covariance singular, so only assert PD when no such augmentation is present.
        if self._n_cumulator_states == 0 and self._n_obs_lag_states == 0:
            P0 = assume(P0, positive_definite=True)
        self.ssm["initial_state_cov"] = P0

    def configure(
        self,
        observed_states: list[str],
        measurement_error: list[str] | None = None,
        constant_params: list[str] | Literal["auto"] | None = None,
        full_shock_covariance: bool = False,
        temporal_aggregation: dict[str, str] | None = None,
        aggregation_period: int = 4,
        ss_obs_intercept: list[str] | None = None,
        observation_equations: dict[str, str] | None = None,
        solver: str = "gensys",
        mode: str | None = None,
        verbose=True,
        max_iter: int = 50,
        tol: float = 1e-6,
        use_adjoint_gradients: bool = True,
        use_direct_lyapunov: bool = False,
    ):
        r"""
        Configure the statespace model for estimation.

        Parameters
        ----------
        observed_states : list of str
            Names of observed series, in data-column order. Each entry is either a
            model variable's ``base_name`` or a key in ``observation_equations``.
        measurement_error : list of str, optional
            Observed states that have measurement error.
        constant_params : list of str or "auto", optional
            Parameters held constant (not estimated). ``"auto"`` freezes all parameters without priors.
        full_shock_covariance : bool
            If True, estimate a full shock covariance matrix instead of diagonal.
        temporal_aggregation : dict of str to str, optional
            Observed states that require temporal aggregation or explicit low-frequency timing.

            - ``"sum"``: Flow variables like GDP (observed = sum of ``aggregation_period`` values).
              Requires cumulator state augmentation.
            - ``"mean"``: Rates or prices reported as period averages. Requires cumulator augmentation.
            - ``"last"``: Point-in-time at end of aggregation window. No cumulator needed.
              Equivalent to omitting the variable, but explicit about timing.
            - ``"first"``: Point-in-time at start of aggregation window. No cumulator needed.
              Data should have values at the first period of each window.

            Variables NOT in this dict use a direct selector — suitable for high-frequency
            observations (no ``NaN`` in data) or low-frequency point-in-time observations.
            The Kalman filter handles missing values automatically.
        aggregation_period : int
            Number of model periods per low-frequency observation. For example, 4 when fitting
            a quarterly model with annual data, or 3 for a monthly model with quarterly data.
            Default is 4.
        ss_obs_intercept : list of str, optional
            Observed states for which to populate ``ssm["obs_intercept"]`` with a
            parameter-dependent steady-state value, re-evaluated on every parameter draw.
            For each entry, the intercept is :math:`\\log v_{ss}(p)` if the variable was
            log-linearized when building the model and :math:`v_{ss}(p)` if it was
            level-linearized. Observed states *not* in this list keep an
            ``obs_intercept`` of zero — appropriate when the data is already in deviation
            form (HP-cycled, demeaned, etc.). Pass ``observed_states`` to enable per-draw
            steady-state subtraction for every observed series. Default ``None`` (no
            entries; ``obs_intercept`` left at zero).
        observation_equations : dict mapping str to str, optional
            Override map keyed by observed-series name (must be a subset of
            ``observed_states``). Values are GCN-syntax expressions in model
            variables and parameters, e.g. ``"log(Y[]) - log(Y[-1]) + log(Z[])"``.
            Contemporaneous and lagged references are accepted; leads are not.
            A name in this dict cannot also appear in ``ss_obs_intercept``.
        solver : str
            Perturbation solver to use.
        mode : str, optional
            PyTensor compilation mode.
        verbose : bool
            Print diagnostic messages.
        max_iter : int
            Maximum iterations for iterative solvers.
        tol : float
            Convergence tolerance for the solver.
        use_adjoint_gradients : bool
            Use adjoint gradients in ``scan_cycle_reduction``.
        use_direct_lyapunov : bool
            Use direct (rather than bilinear) Lyapunov solver.
        """
        # Set up observed states. Names with a user-supplied observation equation
        # are allowed not to correspond to a model state.
        obs_eq_names = set(observation_equations or {})
        unknown_states = [x for x in observed_states if x not in self.state_names and x not in obs_eq_names]
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

        # Validate temporal_aggregation
        if temporal_aggregation is None:
            temporal_aggregation = {}
        else:
            unknown_vars = [x for x in temporal_aggregation if x not in observed_states]
            if unknown_vars:
                raise ValueError(
                    f"The following temporal_aggregation variables are not in observed_states: "
                    f"{', '.join(unknown_vars)}"
                )
            invalid_methods = [
                (var, method) for var, method in temporal_aggregation.items() if method not in VALID_AGGREGATIONS
            ]
            if invalid_methods:
                bad = ", ".join(f"{var}={method!r}" for var, method in invalid_methods)
                raise ValueError(f"Invalid aggregation methods: {bad}. Must be 'sum', 'mean', 'first', or 'last'.")
            has_cumulator_vars = any(m in CUMULATOR_AGGREGATIONS for m in temporal_aggregation.values())
            if has_cumulator_vars and aggregation_period < 2:
                raise ValueError(f"aggregation_period must be >= 2 for sum/mean aggregation, got {aggregation_period}")

        # Validate ss_obs_intercept
        if ss_obs_intercept is None:
            ss_obs_intercept = []
        else:
            unknown_vars = [x for x in ss_obs_intercept if x not in observed_states]
            if unknown_vars:
                raise ValueError(
                    f"The following ss_obs_intercept entries are not in observed_states: {', '.join(unknown_vars)}"
                )
            ss_unknown = [name for name in ss_obs_intercept if not any(v.base_name == name for v in self.variables)]
            if ss_unknown:
                raise ValueError(f"ss_obs_intercept references unknown model variables: {', '.join(ss_unknown)}")

        # Validate observation_equations: keys subset of observed_states, no overlap
        # with ss_obs_intercept. The per-equation symbol resolution happens below
        # when we build the linearization (so we get a single error site).
        if observation_equations is None:
            observation_equations = {}
        else:
            unknown_keys = [k for k in observation_equations if k not in observed_states]
            if unknown_keys:
                raise ValueError(
                    f"The following observation_equations entries are not in observed_states: {', '.join(unknown_keys)}"
                )
            overlap = set(observation_equations) & set(ss_obs_intercept)
            if overlap:
                raise ValueError(
                    f"The following observed states appear in both observation_equations and "
                    f"ss_obs_intercept: {', '.join(sorted(overlap))}. An observation equation "
                    f"already determines its intercept; remove these names from one or the other."
                )
        # Validate constant params
        if constant_params is None:
            constant_params = []
        elif constant_params == "auto":
            param_prior_names = set(self.param_priors.keys())
            constant_params = [x.name for x in self.input_parameters if x.name not in param_prior_names]
        else:
            input_param_names = [x.name for x in self.input_parameters]
            unknown_params = [x for x in constant_params if x not in input_param_names]
            if len(unknown_params) > 0:
                raise ValueError(
                    f"The following parameters are unknown to the model and cannot be set as constant: "
                    f"{', '.join(unknown_params)}"
                )

        # Validate solver argument
        if solver not in VALID_SOLVERS:
            raise ValueError(f"Unknown solver {solver!r}, expected one of {', '.join(repr(s) for s in VALID_SOLVERS)}")

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
        elif solver == "backward_direct":
            solver_kwargs = {}
        else:
            solver_kwargs = {
                "tol": tol,
                "max_iter": max_iter,
                "use_adjoint_gradients": use_adjoint_gradients,
            }

        self._obs_state_names = observed_states
        self.error_states = measurement_error
        self.constant_parameters = constant_params

        self._temporal_aggregation = temporal_aggregation
        self._aggregation_period = aggregation_period
        self._ss_obs_intercept_states = ss_obs_intercept

        # Parse + linearize observation equations now so any errors surface here
        # at configure time rather than mid-graph-build.
        self._obs_equations = {}
        for obs_name, expr_str in observation_equations.items():
            sym = self._parse_observation_equation(obs_name, expr_str)
            intercept_sym, coeffs_sym = self._linearize_observation_equation(sym)
            intercept_pt, coeffs_pt = self._obs_eq_to_pytensor(intercept_sym, coeffs_sym)
            self._obs_equations[obs_name] = (intercept_pt, coeffs_pt)

        # Per-variable lag depth required by each obs equation, including the
        # ``aggregation_period - 1`` headroom needed to broadcast coefficients
        # across the cumulator window when ``sum``/``mean`` aggregation is on.
        # A reference at lag ``-k`` with sum/mean aggregation over ``s`` periods
        # contributes at effective lags ``-k, -k-1, ..., -k-(s-1)``, so the
        # deepest effective lag is ``k + (s-1)``.
        self._obs_lag_depths = {}
        for obs_name, (_intercept_pt, coeffs_pt) in self._obs_equations.items():
            broadcast = aggregation_period - 1 if temporal_aggregation.get(obs_name) in CUMULATOR_AGGREGATIONS else 0
            for vname, lag in coeffs_pt:
                depth_required = -lag + broadcast
                if depth_required > 0:
                    self._obs_lag_depths[vname] = max(self._obs_lag_depths.get(vname, 0), depth_required)

        self.full_covariance = full_shock_covariance
        self.use_direct_lyapunov = use_direct_lyapunov
        self._configured = True
        self._solver = solver
        self._solver_kwargs = solver_kwargs
        self._mode = mode

        # Cumulator-aggregated observed states that are also observation equations
        # route their lag storage through the obs-eq lag block, not the cumulator
        # block — exclude them here to match ``_cumulator_variables``.
        n_cumulator_vars = sum(
            1
            for name, method in temporal_aggregation.items()
            if method in CUMULATOR_AGGREGATIONS and name not in self._obs_equations
        )
        n_cumulator = n_cumulator_vars * (aggregation_period - 1)
        n_obs_lag = sum(self._obs_lag_depths.values())
        k_states_aug = self._k_orig_states + n_cumulator + n_obs_lag

        # Fixed-order layout of the obs-eq lag block: each variable's slots run
        # consecutively, in the dict's insertion order.
        self._obs_lag_starts = {}
        offset = self._k_orig_states + n_cumulator
        for vname, depth in self._obs_lag_depths.items():
            self._obs_lag_starts[vname] = offset
            offset += depth

        super().__init__(
            k_endog,
            k_states_aug,
            self.k_posdef,
            measurement_error=len(measurement_error) > 0,
            verbose=verbose,
        )

        for variable in self.input_parameters:
            if variable.name not in constant_params:
                self._tensor_variable_info = self._tensor_variable_info.add(
                    SymbolicVariable(name=variable.name, symbolic_variable=variable)
                )

    def set_states(self) -> tuple[State, ...]:
        observed_states = self._obs_state_names if self._obs_state_names is not None else []
        hidden_states = [State(name=x.base_name, observed=False) for x in self.variables]
        cumulator_states = [State(name=name, observed=False) for name in self._cumulator_state_names]
        obs_lag_states = [State(name=name, observed=False) for name in self._obs_lag_state_names]
        observed_states = [State(name=name, observed=True) for name in observed_states]
        return *hidden_states, *cumulator_states, *obs_lag_states, *observed_states

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
        solver_tol: float = 1e-8,
    ) -> None:
        super().build_statespace_graph(
            data=data,
            register_data=register_data,
            missing_fill_value=missing_fill_value,
            cov_jitter=cov_jitter,
            save_kalman_filter_outputs_in_idata=save_kalman_filter_outputs_in_idata,
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

        policy_resid, *bk_output = graph_replace(
            [self._policy_resid, *self._bk_output],
            replace=replacement_dict,
            strict=False,
        )

        bk_satisfied, _n_forward, _n_gt_one = bk_output

        if add_norm_check:
            # Diagnostics-only: expose the deterministic and stochastic recursion residuals
            # as Deterministics for posterior inspection. No Potential is added because the
            # solver-convergence check below already gates the logp.
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

            pm.Deterministic("deterministic_norm", pt.linalg.norm(A_prime + B @ R_prime + C @ R_prime @ P))
            pm.Deterministic("stochastic_norm", pt.linalg.norm(B @ S_prime + C @ R_prime @ Q + D))

        if add_bk_check:
            pm.Deterministic("bk_satisfied", bk_satisfied)
            pm.Potential("bk_condition_satisfied", pt.switch(pt.eq(bk_satisfied, 0.0), -np.inf, 0.0))

        if add_solver_success_check:
            pm.Deterministic("policy_resid", policy_resid)
            pm.Potential(
                "policy_resid_within_tol",
                pt.switch(pt.lt(policy_resid, solver_tol), 0.0, -np.inf),
            )

    def sample_autocorrelation_matrices(
        self,
        idata,
        n_lags: int = 10,
        observed: bool = False,
        lag_step: int = 1,
        compile_kwargs: dict | None = None,
    ) -> xr.DataArray:
        r"""
        Posterior distribution of the model-implied autocorrelation matrices.

        For each posterior draw the stationary state covariance :math:`\Sigma` is found from the discrete Lyapunov
        equation :math:`\Sigma = T \Sigma T^\top + R Q R^\top`, and the autocorrelation at lag :math:`k` is
        :math:`T^{k \cdot \texttt{lag\_step}} \Sigma`, normalized by the state standard deviations. The whole
        calculation is built as a single PyTensor graph and evaluated across every draw at once with
        :func:`pymc.compute_deterministics`, so there is no Python loop over samples.

        The model must already have been built with :meth:`build_statespace_graph`, which the presence of ``idata``
        implies.

        Parameters
        ----------
        idata : arviz.InferenceData
            Inference data whose ``posterior`` group holds draws of the model parameters.
        n_lags : int
            Number of non-zero lags to compute; the returned ``lag`` dimension has ``n_lags + 1`` entries. Default 10.
        observed : bool
            Return the autocorrelation of the *observed* series -- the design matrix applied to the state, with
            measurement error included in the lag-0 variance -- instead of the latent states. Default False.
        lag_step : int
            Spacing between lags, in model periods. Use 1 for the model's native frequency. For an observable that is
            a temporal aggregate (for example an annual series from a quarterly model), set this to the aggregation
            period so successive lags are one observation apart. Default 1.
        compile_kwargs : dict, optional
            Passed through to :func:`pymc.compute_deterministics`.

        Returns
        -------
        DataArray
            Autocorrelation matrices with dimensions ``(chain, draw, lag, state, state_aux)``, where the state
            dimensions are the observed states when ``observed`` is True and the latent states otherwise.
        """
        posterior = idata.posterior if hasattr(idata, "posterior") else idata
        state_dim = OBS_STATE_DIM if observed else ALL_STATE_DIM
        aux_dim = OBS_STATE_AUX_DIM if observed else ALL_STATE_AUX_DIM
        name = "observation_autocorrelation" if observed else "autocorrelation"
        coords = {**self.coords, "lag": np.arange(n_lags + 1)}

        with pm.Model(coords=coords) as acf_model:
            self._build_dummy_graph()
            self._insert_random_variables()
            _, _, _, _, T, Z, R, H, Q = self.unpack_statespace()

            Sigma = pt.linalg.solve_discrete_lyapunov(T, R @ Q @ R.T)
            # PD is unsafe when cumulator/obs-lag augmentation makes the stationary covariance singular.
            if self._n_cumulator_states == 0 and self._n_obs_lag_states == 0:
                Sigma = assume(Sigma, positive_definite=True)

            # Advance ``lag_step`` model periods per lag, then accumulate T^(k * lag_step) for k = 0 .. n_lags.
            T_step = T
            for _ in range(lag_step - 1):
                T_step = T_step @ T
            eye = pt.eye(T.shape[0])
            powers = pytensor.scan(
                lambda prev, mat: prev @ mat,
                outputs_info=eye,
                non_sequences=[T_step],
                n_steps=n_lags,
                return_updates=False,
            )
            T_powers = pt.concatenate([eye[None], powers], axis=0)

            if observed:
                autocov = (Z @ (T_powers @ Sigma)) @ Z.T
                autocov_0 = Z @ Sigma @ Z.T + H  # observed lag-0 variance includes measurement error
            else:
                autocov = T_powers @ Sigma
                autocov_0 = Sigma
            autocov = pt.set_subtensor(autocov[0], autocov_0)

            std = pt.sqrt(pt.diag(autocov_0))
            autocorr = autocov / pt.outer(std, std)[None]
            pm.Deterministic(name, autocorr, dims=("lag", state_dim, aux_dim))

        return pm.compute_deterministics(
            posterior, var_names=[name], model=acf_model, compile_kwargs=compile_kwargs, progressbar=False
        )[name]

    def to_pymc(self, exclude_priors: list[str] | None = None):
        if exclude_priors is None:
            exclude_priors = []

        constant_params = self.constant_parameters if self.constant_parameters is not None else []
        skip = set(exclude_priors) | set(constant_params)

        with pm.modelcontext(None):
            for prior, dist in self.param_priors.items():
                if prior in skip:
                    continue
                dist.to_pymc(name=prior)

            for prior, dist in self.shock_priors.items():
                if prior in skip:
                    continue
                dist.to_pymc()


def data_from_prior(
    statepace_mod: DSGEStateSpace,
    pymc_model: pm.Model,
    index: pd.DatetimeIndex | None = None,
    n_samples: int = 500,
    pct_missing: float = 0,
    random_seed: np.random.Generator | int | None = None,
    mvn_method: str = "svd",
    build_statespace_kwargs: dict | None = None,
) -> tuple[xr.Dataset, pd.DataFrame, xr.DataTree]:
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
    mvn_method : str, optional
        Method to use for sampling from the multivariate normal distribution of the state transitions. Passed to
        sample_unconditional_posterior.
    build_statespace_kwargs : dict, optional
        Additional keyword arguments passed to DSGEStateSpace.build_statespace_graph

    Returns
    -------
    true_parameters: xr.Dataset
        True parameters used to generate the data.
    data: pd.DataFrame
        Generated data.
    prior_idata: xr.DataTree
        Draws from the prior predictive distribution, plus conditional prior predictive samples.
    """
    rng = np.random.default_rng(random_seed)
    default_statespace_kwargs = {
        "add_bk_check": False,
        "add_solver_success_check": True,
        "add_norm_check": True,
        "add_steady_state_penalty": True,
    }

    if build_statespace_kwargs is None:
        build_statespace_kwargs = {}

    default_statespace_kwargs.copy().update(build_statespace_kwargs)

    if index is None:
        index = pd.date_range(start="1980-01-01", end="2024-11-01", freq="QS-OCT")
    dummy_data = pd.DataFrame(np.nan, index=index, columns=statepace_mod.observed_states)
    dummy_data.index.freq = dummy_data.index.inferred_freq

    # Copy the model so the original model is unchanged
    new_model = pymc_model.copy()

    with new_model:
        if "data" not in new_model:
            statepace_mod.build_statespace_graph(dummy_data, **build_statespace_kwargs)
        else:
            pm.set_data({"data": dummy_data.fillna(-9999)})

    with warnings.catch_warnings(action="ignore"), freeze_dims_and_data(new_model):
        prior_idata = pm.sample_prior_predictive(
            n_samples, compile_kwargs={"mode": statepace_mod._mode}, random_seed=rng
        )

    with warnings.catch_warnings(action="ignore"):
        prior_trajectories = statepace_mod.sample_unconditional_prior(
            prior_idata, random_seed=rng, mvn_method=mvn_method
        )

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


def prepare_mixed_frequency_data(
    low_freq_data: pd.DataFrame,
    high_freq: str,
    aggregation_period: int = 4,
    observation_position: Literal["first", "last"] = "last",
) -> pd.DataFrame:
    """
    Expand low-frequency data to a high-frequency index for mixed-frequency estimation.

    Each low-frequency value is placed at the first or last high-frequency period within
    its aggregation window, with ``NaN`` at all other periods.  The Kalman filter treats
    ``NaN`` entries as missing observations.

    The flow-vs-stock distinction is irrelevant here — both are placed identically.  The
    distinction only matters in :meth:`DSGEStateSpace.configure`, where ``flow_variables``
    triggers cumulator-state augmentation so the observation equation sums over the window.

    Parameters
    ----------
    low_freq_data : pd.DataFrame
        Observed data at low frequency.  The index should be a ``DatetimeIndex`` at the
        low-frequency periodicity (e.g. annual).  Each column corresponds to an observed
        variable.
    high_freq : str
        Pandas frequency string for the high-frequency (model) periodicity, e.g. ``"QS"``
        for quarterly.
    aggregation_period : int
        Number of high-frequency periods per low-frequency observation.  Default is 4
        (annual from quarterly).
    observation_position : str
        Whether the low-frequency observation corresponds to the ``"first"`` or ``"last"``
        high-frequency period in each window.  Default is ``"last"``.

    Returns
    -------
    pd.DataFrame
        High-frequency DataFrame with ``NaN`` at unobserved periods.

    Examples
    --------
    .. code-block:: python

        import pandas as pd

        annual = pd.DataFrame(
            {"GDP": [100, 110], "R": [0.05, 0.04]},
            index=pd.to_datetime(["2020", "2021"]),
        )
        quarterly = prepare_mixed_frequency_data(annual, high_freq="QS")
    """
    pos_idx = 0 if observation_position == "first" else aggregation_period - 1

    all_columns = list(low_freq_data.columns)
    first_date = low_freq_data.index.min()
    hf_index = pd.date_range(start=first_date, periods=len(low_freq_data) * aggregation_period, freq=high_freq)

    result = pd.DataFrame(np.nan, index=hf_index, columns=all_columns)

    for _, row in low_freq_data.iterrows():
        lf_date = row.name
        window_periods = hf_index[(hf_index >= lf_date)][:aggregation_period]

        if len(window_periods) <= pos_idx:
            continue

        result.loc[window_periods[pos_idx]] = row

    # Trim trailing all-NaN rows beyond the last observation window
    last_obs_idx = result.last_valid_index()
    if last_obs_idx is not None:
        result = result.loc[:last_obs_idx]

    result.index.freq = result.index.inferred_freq
    return result
