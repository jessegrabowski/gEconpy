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
from pymc_extras.statespace.core import dummy_graph
from pymc_extras.statespace.core.properties import Coord, Parameter, Shock, State, SymbolicVariable
from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    JITTER_DEFAULT,
    MISSING_FILL,
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
    """
    A :class:`~pymc_extras.statespace.core.statespace.PyMCStateSpace` model of a linearized DSGE.

    The public constructor is :func:`~gEconpy.model.build.statespace_from_gcn`. Call :meth:`configure` before
    building the statespace graph.

    Parameters
    ----------
    variables : list of TimeAwareSymbol
        Variables in the model.
    shocks : list of TimeAwareSymbol
        Shocks in the model.
    equations : list of sympy expressions
        Equations in the model.
    param_dict : dict mapping str to float
        Default parameter values, as defined in the model file.
    hyper_param_dict : dict mapping str to float
        Default hyperparameter values, as defined in the model file.
    param_priors : SymbolDictionary
        Preliz parameter priors, keyed by parameter name.
    shock_priors : SymbolDictionary
        Preliz shock priors, keyed by shock name.
    parameter_mapping : dict mapping TensorVariable to TensorVariable
        Symbolic function mapping input parameters to the full vector of parameters, including deterministic
        parameters.
    steady_state_mapping : dict mapping TensorVariable to TensorVariable
        Symbolic function mapping input parameters to the steady-state values of the model.
    linearized_system : list of TensorVariable
        Four symbolic expressions representing the linearized system of equations as partial jacobians of the
        model equations with respect to variables at time t+1 (A), t (B), t-1 (C), and with respect to exogenous
        shocks (D), each evaluated at the symbolic steady state.
    var_order : ndarray, optional
        Variable column permutation applied when the system was linearized. Defaults to the identity
        permutation.
    log_linearized_variables : list of str, optional
        Base names of variables that were log-linearized when building ``linearized_system``. The
        ``ss_obs_intercept`` option of :meth:`configure` uses it to decide whether an intercept entry is
        ``log(v_ss(p))`` or ``v_ss(p)``. Defaults to no variables.
    sympytensor_cache : dict, optional
        Sympytensor cache mapping cache keys to pytensor nodes, shared with the graphs that built the model.
        Defaults to a new, empty cache.
    filter_type : str, optional
        Kalman filter implementation used for the likelihood. Defaults to "standard".
    mode : str, optional
        PyTensor compilation mode for post-estimation sampling functions. Defaults to None.
    cov_jitter : float, optional
        Jitter added to the diagonal of covariance matrices inside the Kalman filter. Defaults to
        ``JITTER_DEFAULT`` from pymc-extras.
    missing_fill_value : float, optional
        Sentinel that replaces missing observations before the filter runs. Defaults to ``MISSING_FILL`` from
        pymc-extras.
    verbose : bool, optional
        If True, show diagnostic messages. Defaults to True.
    """

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
        mode: str | None = None,
        cov_jitter: float = JITTER_DEFAULT,
        missing_fill_value: float = MISSING_FILL,
        verbose: bool = True,
    ):
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

        # ``linearize_model`` permutes the variable columns to expose the block-zero structure of A and C. The solver
        # returns T and R in that permuted order, and ``_setup_policy_matrices`` applies ``inv_var_order`` to put
        # them back into the user's variable order before they reach the Kalman filter.
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
        self._graph_checks: dict = {}
        self._linearized_system_subbed: list | None = None
        self._policy_graph: list | None = None

        self._bk_output = None
        self._policy_resid = None
        self._n_steps = None
        self._lead_var_idx: np.ndarray | None = None

        self._temporal_aggregation: dict[str, str] = {}
        self._aggregation_period: int = 4
        self._k_orig_states: int = len(variables)

        self._log_linearized_variables: set[str] = set(log_linearized_variables or [])
        self._ss_obs_intercept_states: list[str] = []

        # Sharing the cache with ``statespace_from_gcn`` makes later sympy-to-pytensor conversions (the observation
        # equations) resolve to the same TensorVariables that key ``steady_state_mapping`` and ``parameter_mapping``.
        self._sympytensor_cache: dict = sympytensor_cache if sympytensor_cache is not None else {}

        # Linearized observation equations, keyed by observed-series name, as
        # ``(intercept, {(variable_base_name, lag): coefficient})`` pytensor expressions in the input parameters.
        self._obs_equations: dict[str, tuple] = {}
        self._obs_lag_depths: dict[str, int] = {}
        self._obs_lag_starts: dict[str, int] = {}

        self.verbose = verbose

        k_endog = 1
        k_states = len(variables)
        k_posdef = len(shocks)

        super().__init__(
            k_endog,
            k_states,
            k_posdef,
            filter_type=filter_type,
            verbose=False,
            measurement_error=False,
            mode=mode,
            cov_jitter=cov_jitter,
            missing_fill_value=missing_fill_value,
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
            T, R, n_steps = scan_cycle_reduction(A, B, C, D, **self._solver_kwargs)
            self._n_steps = n_steps

        # A, B, and C have columns in ``var_order`` and T shares that basis until it is remapped below, so the
        # residual has to be evaluated before the remap or the two orderings mix and inflate it.
        resid = pt.square(A + B @ T + C @ T @ T).sum()

        # T comes back in the permuted variable order on both axes and R on its rows.
        if not np.array_equal(self.var_order, np.arange(len(self.var_order))):
            inv = self.inv_var_order
            T = T[inv][:, inv]
            R = R[inv]

        return T, R, resid

    @property
    def lead_var_idx(self) -> np.ndarray:
        """Column indices of forward-looking variables (variables appearing at t+1 in any equation)."""
        if self._lead_var_idx is None:
            idx = [i for i, v in enumerate(self.variables) if any(eq.has(v.set_t(1)) for eq in self.equations)]
            self._lead_var_idx = np.array(idx, dtype=int)
        return self._lead_var_idx

    @property
    def n_forward(self) -> int:
        """Number of forward-looking variables."""
        return len(self.lead_var_idx)

    def _setup_state_covariance(self) -> pt.TensorVariable:
        """Register the shock covariance parameters, store ``state_cov`` in ``ssm``, and return it."""
        if self.full_covariance:
            state_cov = self.make_and_register_variable("state_cov", shape=(self.k_posdef, self.k_posdef))
            Q = assume(state_cov, positive_definite=True)
            self.ssm["state_cov"] = Q
            return Q

        # AssumptionFeature tags ``pt.diag(stack(...))`` as diagonal and propagates symmetric/PSD through the
        # congruence R Q R', so the compiled logp and its gradient can use Cholesky-based solves.
        sigmas = [self.make_and_register_variable(f"sigma_{shock.base_name}", shape=()) for shock in self.shocks]
        Q = pt.diag(pt.stack([s**2 for s in sigmas]))
        self.ssm["state_cov"] = Q
        return Q

    def _selector_columns(self, name: str) -> tuple[list[int], float]:
        """
        Locate the state columns an observed model variable loads on, with the weight each receives.

        A directly observed variable loads on its own column with unit weight. A ``sum`` or ``mean`` aggregated
        variable also loads on its cumulator slots, with weight :math:`1/s` for ``mean`` over an aggregation
        period of :math:`s`.
        """
        columns = [self._orig_state_names.index(name)]
        agg_method = self._temporal_aggregation.get(name)
        if agg_method not in CUMULATOR_AGGREGATIONS:
            return columns, 1.0

        n_cum_lags = self._aggregation_period - 1
        cum_start = self._k_orig_states + self._cumulator_variables.index(name) * n_cum_lags
        columns += list(range(cum_start, cum_start + n_cum_lags))
        weight = 1.0 / self._aggregation_period if agg_method == "mean" else 1.0
        return columns, weight

    def _make_design_matrix(self) -> np.ndarray | pt.TensorVariable:
        """
        Build the observation design matrix :math:`Z`.

        Each observed state either has a user-supplied observation equation, whose linearized coefficients depend
        on the parameters, or is a plain selector of a model variable, weighted across its cumulator slots when
        temporally aggregated.

        Returns
        -------
        Z : ndarray or TensorVariable
            Constant ``(k_endog, k_states)`` array when every observed state is a selector, or a pytensor matrix of
            the same shape when any observation equation contributes parameter-dependent coefficients.
        """
        if not self._obs_equations:
            Z = np.zeros((self.k_endog, self.k_states))
            for i, name in enumerate(self.observed_states):
                columns, weight = self._selector_columns(name)
                Z[i, columns] = weight
            return Z

        Z = pt.zeros((self.k_endog, self.k_states), dtype=floatX)
        for i, name in enumerate(self.observed_states):
            if name not in self._obs_equations:
                columns, weight = self._selector_columns(name)
                Z = Z[i, columns].set(weight)
                continue

            agg_method = self._temporal_aggregation.get(name)
            if agg_method in CUMULATOR_AGGREGATIONS:
                n_periods = self._aggregation_period
                coeff_weight = 1.0 if agg_method == "sum" else 1.0 / n_periods
            else:
                n_periods = 1
                coeff_weight = 1.0

            # A coefficient at lag k broadcast across the aggregation window lands at effective lags k, k-1, ...,
            # so ``inc_subtensor`` lets overlapping contributions accumulate (annual sums of quarterly
            # log-differences telescope this way).
            _, coeffs = self._obs_equations[name]
            for (variable_name, lag), coeff in coeffs.items():
                for d in range(n_periods):
                    effective_lag = lag - d
                    if effective_lag == 0:
                        col = self._orig_state_names.index(variable_name)
                    else:
                        col = self._obs_lag_column(variable_name, effective_lag)
                    Z = Z[i, col].inc(coeff_weight * coeff)
        return Z

    def _make_obs_intercept(self) -> pt.TensorVariable:
        r"""
        Build the observation intercept vector :math:`d`.

        An observed state with a user-supplied observation equation takes the linearization's constant term, for
        example :math:`\log Y_{ss}(p) + \log Z_{ss}(p)` for a growth-rate observation. A state listed in
        ``ss_obs_intercept`` takes :math:`\log v_{ss}(p)` when log-linearized and :math:`v_{ss}(p)` otherwise.
        Every other state takes zero, which is the right intercept for data already in deviation form.
        ``sum``-aggregated observations multiply the per-period intercept by the aggregation period.

        Returns
        -------
        d : TensorVariable
            Vector of ``k_endog`` intercepts in the model's input parameters.
        """
        steady_state_by_name = {k.name: v for k, v in self.steady_state_mapping.items()}
        ss_intercept_states = set(self._ss_obs_intercept_states)

        entries: list[pt.TensorVariable] = []
        for name in self.observed_states:
            if name in self._obs_equations:
                per_period_intercept, _ = self._obs_equations[name]
            elif name in ss_intercept_states:
                ss_key = f"{name}_ss"
                if ss_key not in steady_state_by_name:
                    raise ValueError(
                        f"ss_obs_intercept requested for {name!r}, but no symbolic steady state is available for "
                        f"it. Give {name!r} an analytic steady state in the GCN file, or remove it from tryreduce."
                    )
                v_ss = steady_state_by_name[ss_key]
                per_period_intercept = pt.log(v_ss) if name in self._log_linearized_variables else v_ss
            else:
                entries.append(pt.zeros((), dtype=floatX))
                continue

            if self._temporal_aggregation.get(name) == "sum":
                entries.append(self._aggregation_period * per_period_intercept)
            else:
                entries.append(per_period_intercept)

        return pt.stack(entries).astype(floatX)

    def _parse_observation_equation(self, name: str, expr_str: str) -> sp.Expr:
        """
        Parse a GCN-syntax observation equation into a sympy expression in the model's namespace.

        Contemporaneous and lagged variable references (``v[]``, ``v[-1]``, ...) are accepted. Leads raise, because
        an observation cannot depend on the future.

        Parameters
        ----------
        name : str
            Observed-series name the equation belongs to, used in error messages.
        expr_str : str
            GCN-syntax expression in terms of model variables and parameters.

        Returns
        -------
        expression : sympy expression
            Parsed expression whose free symbols are the model's own variable and parameter symbols.
        """
        ast = parse_expression(expr_str, context=f"observation_equations[{name!r}]")

        # Sympy equality and hashing include assumptions, so the parsed symbols must carry the model variables'
        # assumptions or the linearization's ``xreplace`` silently leaves them in place.
        assumptions = {v.base_name: dict(v.assumptions0) for v in self.variables}
        expression = ast_to_sympy(ast, assumptions=assumptions)

        var_names = {v.base_name for v in self.variables}
        param_names = set(self.param_dict) | set(self.hyper_param_dict)

        for symbol in expression.free_symbols:
            if not isinstance(symbol, TimeAwareSymbol):
                if symbol.name not in param_names:
                    raise ValueError(
                        f"Observation equation {name!r} references unknown symbol {symbol.name!r}: not a model "
                        f"variable, parameter, or hyperparameter."
                    )
                continue

            if symbol.time_index == "ss":
                continue
            if symbol.time_index > 0:
                raise ValueError(
                    f"Observation equation {name!r} contains a lead reference {symbol}. Only contemporaneous and "
                    f"lagged model variables are allowed."
                )
            if symbol.base_name not in var_names:
                raise ValueError(
                    f"Observation equation {name!r} references unknown model variable {symbol.base_name!r}. "
                    f"Known: {sorted(var_names)}"
                )

        return expression

    def _linearize_observation_equation(self, expression: sp.Expr) -> tuple[sp.Expr, dict[tuple[str, int], sp.Expr]]:
        r"""
        Linearize an observation equation to first order around the model's steady state.

        Each variable reference :math:`v_{t+k}` in ``expression`` (with :math:`k \le 0`) is replaced by
        :math:`v_{ss} \exp(\tilde v_k)` for a log-linearized variable and by :math:`v_{ss} + \tilde v_k` for a
        level-linearized one, where :math:`\tilde v_k` is a fresh dummy. The intercept is the value at all
        :math:`\tilde v_k = 0`, and the coefficient on each :math:`\tilde v_k` is the first partial derivative
        there. For a log-linearized :math:`v` the chain rule makes that coefficient
        :math:`v_{ss}\, \partial g / \partial v_{t+k}\big|_{ss}`.

        Parameters
        ----------
        expression : sympy expression
            Observation equation in raw form, in the model's symbol namespace.

        Returns
        -------
        intercept : sympy expression
            Constant term :math:`g(x_{ss}, p)` in steady-state symbols and parameters.
        coeffs : dict mapping (str, int) to sympy expression
            Linear coefficient of each appearing ``(variable_base_name, time_index)`` pair, where ``time_index`` is
            0 for contemporaneous references and negative for lags.
        """
        var_by_name = {v.base_name: v for v in self.variables}
        appearing = {
            (symbol.base_name, symbol.time_index)
            for symbol in expression.free_symbols
            if isinstance(symbol, TimeAwareSymbol) and symbol.time_index != "ss"
        }

        deviation_form: dict[TimeAwareSymbol, sp.Expr] = {}
        deviations: dict[tuple[str, int], sp.Symbol] = {}
        for variable_name, lag in appearing:
            variable = var_by_name[variable_name]
            v_at_t = variable.set_t(lag)
            v_ss = variable.set_t("ss")
            lag_tag = "0" if lag == 0 else f"m{-lag}"
            deviation = sp.Symbol(f"_tilde_{variable_name}_{lag_tag}", real=True)
            deviations[(variable_name, lag)] = deviation
            if variable_name in self._log_linearized_variables:
                deviation_form[v_at_t] = v_ss * sp.exp(deviation)
            else:
                deviation_form[v_at_t] = v_ss + deviation

        g = expression.xreplace(deviation_form)
        at_steady_state = {deviation: sp.Integer(0) for deviation in deviations.values()}

        intercept = g.xreplace(at_steady_state)
        coeffs = {key: sp.diff(g, deviation).xreplace(at_steady_state) for key, deviation in deviations.items()}

        return intercept, coeffs

    def _obs_eq_to_pytensor(
        self, intercept_sym: sp.Expr, coeffs_sym: dict[tuple[str, int], sp.Expr]
    ) -> tuple[pt.TensorVariable, dict[tuple[str, int], pt.TensorVariable]]:
        """
        Convert a linearized observation equation to pytensor expressions in the model's input parameters.

        ``self._sympytensor_cache`` makes the steady-state and parameter symbols resolve to the same TensorVariables
        that key ``self.steady_state_mapping`` and ``self.parameter_mapping``, so a ``graph_replace`` with the
        steady-state mapping then swaps each steady-state symbol for its expression in the input parameters.

        Parameters
        ----------
        intercept_sym : sympy expression
            Constant term of the linearization.
        coeffs_sym : dict mapping (str, int) to sympy expression
            Coefficient of each appearing ``(variable, lag)`` pair.

        Returns
        -------
        intercept : TensorVariable
            Scalar intercept in the input parameters.
        coeffs : dict mapping (str, int) to TensorVariable
            Scalar coefficient in the input parameters, one per ``(variable, lag)`` pair.
        """

        def to_tensor(sym_expr: sp.Expr) -> pt.TensorVariable:
            # Sympy integer constants such as 0 or 1 come through as Python ints.
            tensor = as_tensor(sym_expr, self._sympytensor_cache)
            if not isinstance(tensor, pt.Variable):
                tensor = pt.as_tensor_variable(tensor)
            return pt.cast(graph_replace(tensor, self.steady_state_mapping, strict=False), floatX)

        intercept = to_tensor(intercept_sym)
        coeffs = {key: to_tensor(coeff_sym) for key, coeff_sym in coeffs_sym.items()}
        return intercept, coeffs

    @property
    def _n_cumulator_states(self) -> int:
        return len(self._cumulator_variables) * (self._aggregation_period - 1)

    @property
    def _cumulator_variables(self) -> list[str]:
        # An aggregated observation equation stores its lags in the observation-lag block, because its series name
        # need not be a model variable, so it is not a cumulator variable.
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
        """Return the augmented-state column holding ``var_name`` at negative time index ``lag``."""
        depth = -lag
        return self._obs_lag_starts[var_name] + (depth - 1)

    def _augment_transition(self, T: pt.TensorVariable) -> pt.TensorVariable:
        """
        Augment the transition matrix with cumulator rows/columns for temporally aggregated variables.

        The augmented matrix has the block form::

            T_aug = [ T  |  0              ]
                    [----|-----------------|
                    [ F  |  kron(I_n, C)   ]

        where ``C`` is the constant ``(s-1) x (s-1)`` lower-shift companion matrix shared by all aggregated
        variables, and ``F`` holds unit selectors that copy each aggregated variable into the first cumulator slot.

        Parameters
        ----------
        T : TensorVariable
            Transition matrix of shape ``(k_orig, k_orig)`` from the perturbation solution.

        Returns
        -------
        T_aug : TensorVariable
            Augmented transition matrix of shape ``(k_orig + n_cum, k_orig + n_cum)``.
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

        # ``block`` lets ``local_block_dot_to_block_of_dots`` split a downstream ``T_aug @ x`` into block products and
        # drop the zero top-right block entirely.
        zero_block = pt.zeros((k_orig, n_cum), dtype=floatX)
        return block(
            [
                [T, zero_block],
                [F, pt.constant(C)],
            ]
        )

    def _append_obs_lag_block(self, T_aug: pt.TensorVariable) -> pt.TensorVariable:
        """
        Append shift-companion chains for variables that observation equations reference at a lag.

        Each variable :math:`v` with required lag depth :math:`d` gets :math:`d` slots. Slot 1 copies :math:`v`
        from the previous period and each later slot copies the slot before it.

        Parameters
        ----------
        T_aug : TensorVariable
            Augmented transition matrix with the cumulator block already appended.

        Returns
        -------
        T_aug : TensorVariable
            The same matrix with the observation-lag block appended in the trailing rows and columns.
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
        Append zero rows to the selection matrix for the cumulator and observation-lag states.

        The augmented matrix has the block form::

            R_aug = [ R ]
                    [---]
                    [ 0 ]

        The extra rows are zero because the appended states are deterministic lag copies.

        Parameters
        ----------
        R : TensorVariable
            Selection matrix of shape ``(k_orig, k_posdef)``.

        Returns
        -------
        R_aug : TensorVariable
            Augmented selection matrix of shape ``(k_orig + n_extra, k_posdef)``.
        """
        n_extra = self._n_cumulator_states + self._n_obs_lag_states
        if n_extra == 0:
            return R

        zeros = pt.zeros((n_extra, self.k_posdef), dtype=floatX)
        return pt.join(-2, R, zeros)

    def make_symbolic_graph(self):
        """
        Build the symbolic statespace matrices of the linearized DSGE model into ``ssm``.

        The transition and selection matrices come from the perturbation solution, augmented with any cumulator
        and observation-lag states. The design matrix, intercept, and covariances follow from the options passed
        to :meth:`configure`, which has to run first.
        """
        if not self._configured:
            if self.verbose:
                _log.info("Statespace model construction complete, but call the .configure method to finalize.")
            return

        constant_replacements = {
            parameter: pt.constant(np.array(self.param_dict[parameter.name]).astype(floatX), name=parameter.name)
            for parameter in self.input_parameters
            if parameter.name in self.constant_parameters
        }

        self._linearized_system_subbed = [A, B, C, D] = graph_replace(
            self.linearized_system, constant_replacements, strict=False
        )

        # Constants left as free inputs of the observation equations would have no PyMC variable to bind to when the
        # logp is compiled.
        if constant_replacements and self._obs_equations:
            self._obs_equations = {
                name: (
                    graph_replace(intercept_pt, constant_replacements, strict=False),
                    {v: graph_replace(c, constant_replacements, strict=False) for v, c in coeffs_pt.items()},
                )
                for name, (intercept_pt, coeffs_pt) in self._obs_equations.items()
            }

        # A, B, and C have columns in ``var_order``, so ``lead_var_idx`` has to move to the permuted positions.
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
            # The ``ss_obs_intercept`` branch reads ``steady_state_mapping`` directly, which is in the free
            # parameter placeholders, so the constants have to be substituted here as well.
            if constant_replacements:
                obs_intercept = graph_replace(obs_intercept, constant_replacements, strict=False)
            self.ssm["obs_intercept"] = obs_intercept

        Q = self._setup_state_covariance()

        if self.measurement_error:
            sigmas = [self.make_and_register_variable(f"error_sigma_{state}", shape=()) for state in self.error_states]
            variances = pt.stack([sigma**2 for sigma in sigmas])
            if list(self.error_states) == list(self.observed_states):
                error_variances = variances
            else:
                error_positions = [self.observed_states.index(state) for state in self.error_states]
                error_variances = pt.zeros((self.k_endog,), dtype=floatX)[error_positions].set(variances)
            self.ssm["obs_cov"] = pt.diag(error_variances)

        self.ssm["initial_state"] = pt.zeros(self.k_states)

        method = "direct" if self.use_direct_lyapunov else "bilinear"
        P0 = pt.linalg.solve_discrete_lyapunov(T_aug, R_aug @ Q @ R_aug.T, method=method)
        # Deterministic cumulator and observation-lag copies make the stationary covariance singular.
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
        verbose: bool = True,
        max_iter: int = 50,
        tol: float = 1e-6,
        use_adjoint_gradients: bool = True,
        use_direct_lyapunov: bool = False,
    ) -> None:
        r"""
        Choose the observed series, estimated parameters, and solver, then size the statespace model to match.

        Parameters
        ----------
        observed_states : list of str
            Names of observed series, in data-column order. Each entry is either a model variable's ``base_name``
            or a key in ``observation_equations``.
        measurement_error : list of str, optional
            Observed states that have measurement error. Defaults to none.
        constant_params : list of str or "auto", optional
            Parameters held at their GCN values and excluded from estimation. ``"auto"`` freezes every parameter
            without a prior. Defaults to none.
        full_shock_covariance : bool, optional
            Estimate a full shock covariance matrix. The default estimates only the diagonal. Defaults to False.
        temporal_aggregation : dict mapping str to str, optional
            Observed states that are aggregated over a low-frequency window, with the aggregation method.
            ``"sum"`` observes the sum of ``aggregation_period`` model periods (flow variables such as GDP) and
            ``"mean"`` observes their average (rates and prices reported as period averages). Both add cumulator
            states. ``"last"`` and ``"first"`` observe the model variable at the end or start of the window, add
            no states, and differ from omitting the variable only by making the timing explicit. States not in
            this dict use a direct selector, which suits high-frequency series and low-frequency point-in-time
            series alike, since the Kalman filter treats ``NaN`` as missing. Defaults to none.
        aggregation_period : int, optional
            Number of model periods per low-frequency observation, for example 4 for a quarterly model with
            annual data or 3 for a monthly model with quarterly data. Defaults to 4.
        ss_obs_intercept : list of str, optional
            Observed states whose ``obs_intercept`` entry is the parameter-dependent steady state, re-evaluated on
            every draw. The entry is :math:`\log v_{ss}(p)` for a log-linearized variable and :math:`v_{ss}(p)`
            for a level-linearized one. Every other observed state keeps an intercept of zero, which is right for
            data already in deviation form. Pass ``observed_states`` to subtract the steady state from every
            series. Defaults to none.
        observation_equations : dict mapping str to str, optional
            GCN-syntax expressions in model variables and parameters, keyed by observed-series name, for example
            ``"log(Y[]) - log(Y[-1]) + log(Z[])"``. Every key must appear in ``observed_states`` and none may
            appear in ``ss_obs_intercept``, because an observation equation fixes its own intercept.
            Contemporaneous and lagged references are accepted, leads are not. Defaults to none.
        solver : str, optional
            Perturbation solver, one of ``"gensys"``, ``"cycle_reduction"``, ``"scan_cycle_reduction"``, or
            ``"backward_direct"``. Defaults to ``"gensys"``.
        mode : str, optional
            PyTensor compilation mode for post-estimation sampling functions. Defaults to None.
        verbose : bool, optional
            Print diagnostic messages. Defaults to True.
        max_iter : int, optional
            Maximum iterations for the iterative solvers. Defaults to 50.
        tol : float, optional
            Convergence tolerance for the solver. Defaults to 1e-6.
        use_adjoint_gradients : bool, optional
            Differentiate ``scan_cycle_reduction`` with the adjoint method. Defaults to True.
        use_direct_lyapunov : bool, optional
            Solve the initial-state Lyapunov equation with the direct method. The default uses the bilinear method.
            Defaults to False.

        Examples
        --------
        With ``constant_params="auto"``, every parameter without a prior is frozen and the rest are estimated:

        .. code-block:: python

            from gEconpy import statespace_from_gcn
            from gEconpy.data import get_example_gcn

            ss_mod = statespace_from_gcn(get_example_gcn("RBC"), verbose=False)
            ss_mod.configure(
                observed_states=["Y", "C"],
                measurement_error=["Y", "C"],
                constant_params="auto",
                verbose=False,
            )
            print(ss_mod.param_names)

        Summing ``Y`` over each year adds three cumulator states to a quarterly model, while ``C`` stays quarterly:

        .. code-block:: python

            from gEconpy import statespace_from_gcn
            from gEconpy.data import get_example_gcn

            ss_mod = statespace_from_gcn(get_example_gcn("RBC"), verbose=False)
            ss_mod.configure(
                observed_states=["Y", "C"],
                measurement_error=["Y", "C"],
                temporal_aggregation={"Y": "sum"},
                aggregation_period=4,
                verbose=False,
            )
            print(ss_mod.state_names)
        """
        obs_eq_names = set(observation_equations or {})
        unknown_states = [x for x in observed_states if x not in self.state_names and x not in obs_eq_names]
        if unknown_states:
            raise ValueError(
                f"The following states are unknown to the model and cannot be set as observed: "
                f"{', '.join(unknown_states)}"
            )

        if measurement_error is None:
            measurement_error = []
        unobserved_error_states = [x for x in measurement_error if x not in observed_states]
        if unobserved_error_states:
            raise ValueError(
                f"The following states are not observed, and cannot have measurement error: "
                f"{', '.join(unobserved_error_states)}"
            )

        if temporal_aggregation is None:
            temporal_aggregation = {}
        self._validate_temporal_aggregation(temporal_aggregation, aggregation_period, observed_states)

        if ss_obs_intercept is None:
            ss_obs_intercept = []
        unobserved_intercepts = [x for x in ss_obs_intercept if x not in observed_states]
        if unobserved_intercepts:
            raise ValueError(
                f"The following ss_obs_intercept entries are not in observed_states: {', '.join(unobserved_intercepts)}"
            )
        unknown_intercepts = [name for name in ss_obs_intercept if name not in self._orig_state_names]
        if unknown_intercepts:
            raise ValueError(f"ss_obs_intercept references unknown model variables: {', '.join(unknown_intercepts)}")

        if observation_equations is None:
            observation_equations = {}
        unobserved_equations = [k for k in observation_equations if k not in observed_states]
        if unobserved_equations:
            raise ValueError(
                f"The following observation_equations entries are not in observed_states: "
                f"{', '.join(unobserved_equations)}"
            )
        overlap = set(observation_equations) & set(ss_obs_intercept)
        if overlap:
            raise ValueError(
                f"The following observed states appear in both observation_equations and ss_obs_intercept: "
                f"{', '.join(sorted(overlap))}. An observation equation already determines its intercept, so "
                f"remove these names from one or the other."
            )

        if constant_params is None:
            constant_params = []
        elif constant_params == "auto":
            constant_params = [x.name for x in self.input_parameters if x.name not in self.param_priors]
        else:
            input_param_names = [x.name for x in self.input_parameters]
            unknown_params = [x for x in constant_params if x not in input_param_names]
            if unknown_params:
                raise ValueError(
                    f"The following parameters are unknown to the model and cannot be set as constant: "
                    f"{', '.join(unknown_params)}"
                )

        if solver not in VALID_SOLVERS:
            raise ValueError(f"Unknown solver {solver!r}, expected one of {', '.join(repr(s) for s in VALID_SOLVERS)}")

        k_endog = len(observed_states)
        n_stochastic_sources = len(measurement_error) + len(self.shock_names)
        if k_endog > n_stochastic_sources:
            verb = "are" if n_stochastic_sources != 1 else "is"
            suffix = "s" if n_stochastic_sources != 1 else ""
            raise ValueError(
                f"Stochastic singularity! You requested {k_endog} observed timeseries, but there {verb} "
                f"only {n_stochastic_sources} source{suffix} of stochastic variation. "
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
            solver_kwargs = {"tol": tol, "max_iter": max_iter, "use_adjoint_gradients": use_adjoint_gradients}

        self._obs_state_names = observed_states
        self.error_states = measurement_error
        self.constant_parameters = constant_params

        self._temporal_aggregation = temporal_aggregation
        self._aggregation_period = aggregation_period
        self._ss_obs_intercept_states = ss_obs_intercept

        # Parsing and linearizing here surfaces a malformed observation equation at configure time, before the graph
        # build starts.
        self._obs_equations = {}
        for obs_name, expr_str in observation_equations.items():
            expression = self._parse_observation_equation(obs_name, expr_str)
            intercept_sym, coeffs_sym = self._linearize_observation_equation(expression)
            self._obs_equations[obs_name] = self._obs_eq_to_pytensor(intercept_sym, coeffs_sym)

        # A reference at lag k in an observation equation aggregated over s periods contributes at effective lags
        # k, k+1, ..., k+(s-1), so the lag chain needs s-1 extra slots of headroom.
        self._obs_lag_depths = {}
        for obs_name, (_intercept, coeffs) in self._obs_equations.items():
            headroom = aggregation_period - 1 if temporal_aggregation.get(obs_name) in CUMULATOR_AGGREGATIONS else 0
            for variable_name, lag in coeffs:
                depth_required = -lag + headroom
                if depth_required > 0:
                    current_depth = self._obs_lag_depths.get(variable_name, 0)
                    self._obs_lag_depths[variable_name] = max(current_depth, depth_required)

        self.full_covariance = full_shock_covariance
        self.use_direct_lyapunov = use_direct_lyapunov
        self._configured = True
        self._solver = solver
        self._solver_kwargs = solver_kwargs

        n_cumulator = self._n_cumulator_states
        n_obs_lag = self._n_obs_lag_states
        k_states_aug = self._k_orig_states + n_cumulator + n_obs_lag

        # Each variable's observation-lag slots run consecutively, in insertion order of ``_obs_lag_depths``.
        self._obs_lag_starts = {}
        offset = self._k_orig_states + n_cumulator
        for variable_name, depth in self._obs_lag_depths.items():
            self._obs_lag_starts[variable_name] = offset
            offset += depth

        super().__init__(
            k_endog,
            k_states_aug,
            self.k_posdef,
            filter_type=self.filter_type,
            measurement_error=len(measurement_error) > 0,
            mode=mode,
            cov_jitter=self.cov_jitter,
            missing_fill_value=self.missing_fill_value,
            verbose=verbose,
        )

        for variable in self.input_parameters:
            if variable.name not in constant_params:
                self._tensor_variable_info = self._tensor_variable_info.add(
                    SymbolicVariable(name=variable.name, symbolic_variable=variable)
                )

    @staticmethod
    def _validate_temporal_aggregation(
        temporal_aggregation: dict[str, str], aggregation_period: int, observed_states: list[str]
    ) -> None:
        unobserved = [x for x in temporal_aggregation if x not in observed_states]
        if unobserved:
            raise ValueError(
                f"The following temporal_aggregation variables are not in observed_states: {', '.join(unobserved)}"
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

    def set_states(self) -> tuple[State, ...]:
        """
        List the states of the statespace model.

        Returns
        -------
        states : tuple
            Hidden model variables, cumulator states, observation lag states, and observed states, in that order.
        """
        observed_names = self._obs_state_names if self._obs_state_names is not None else []
        hidden_states = [State(name=x.base_name, observed=False) for x in self.variables]
        cumulator_states = [State(name=name, observed=False) for name in self._cumulator_state_names]
        obs_lag_states = [State(name=name, observed=False) for name in self._obs_lag_state_names]
        observed_states = [State(name=name, observed=True) for name in observed_names]
        return *hidden_states, *cumulator_states, *obs_lag_states, *observed_states

    def set_parameters(self) -> tuple[Parameter, ...]:
        """
        List the parameters the statespace model expects from the PyMC model.

        Returns
        -------
        parameters : tuple of Parameter
            Non-constant model parameters, followed by the shock covariance parameters and any measurement error
            parameters.
        """
        parameters = [
            Parameter(name=x.name, shape=()) for x in self.input_parameters if x.name not in self.constant_parameters
        ]

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

    def set_shocks(self) -> tuple[Shock, ...]:
        """
        List the shocks of the statespace model.

        Returns
        -------
        shocks : tuple
            One shock per exogenous shock in the model.
        """
        return tuple(Shock(name=x.base_name) for x in self.shocks)

    def set_coords(self) -> tuple[Coord, ...]:
        """
        List the coordinates of the statespace model.

        Returns
        -------
        coords : tuple
            The default coordinates implied by the model states, shocks, and observed states.
        """
        return self.default_coords()

    @property
    def param_dims(self) -> dict[str, tuple[str, ...] | None]:
        """Dimension names of each model parameter, empty until the model is configured."""
        if not self._configured:
            return {}

        return {param: None if param != "state_cov" else (SHOCK_DIM, SHOCK_AUX_DIM) for param in self.param_names}

    def build_statespace_graph(
        self,
        data: np.ndarray | pd.DataFrame | pt.TensorVariable,
        add_norm_check: bool = True,
        add_bk_check: bool = False,
        add_solver_success_check: bool = False,
        solver_tol: float = 1e-8,
    ) -> None:
        """
        Build the Kalman filter likelihood for ``data`` into the active PyMC model, plus DSGE diagnostics.

        Rebuilding into a model that already holds this graph only repoints it at ``data``. The diagnostic flags
        are read on the first build and ignored afterwards.

        Parameters
        ----------
        data : numpy array, pandas DataFrame, or pytensor tensor
            Observed data to fit against. Missing values are filled with ``missing_fill_value`` and marginalized
            by the filter.
        add_norm_check : bool, optional
            Register the deterministic and stochastic recursion residual norms as Deterministics. Defaults to True.
        add_bk_check : bool, optional
            Register the Blanchard-Kahn indicator and a Potential that rejects draws violating it. Defaults to
            False.
        add_solver_success_check : bool, optional
            Register the policy-function residual and a Potential that rejects draws where it exceeds
            ``solver_tol``. Defaults to False.
        solver_tol : float, optional
            Residual tolerance used by ``add_solver_success_check``. Defaults to 1e-8.

        Examples
        --------
        With the GCN priors and the shock scale registered in the model context, the likelihood attaches to the
        observed output:

        .. code-block:: python

            import numpy as np
            import pandas as pd
            import pymc as pm

            from gEconpy import statespace_from_gcn
            from gEconpy.data import get_example_gcn

            ss_mod = statespace_from_gcn(get_example_gcn("RBC"), verbose=False)
            ss_mod.configure(observed_states=["Y"], constant_params="auto", verbose=False)

            index = pd.date_range("2000-01-01", periods=40, freq="QS")
            data = pd.DataFrame(np.random.default_rng(0).normal(scale=0.01, size=(40, 1)), index=index, columns=["Y"])

            with pm.Model(coords=ss_mod.coords) as pm_mod:
                ss_mod.to_pymc()
                pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)
                ss_mod.build_statespace_graph(data)
                print(pm_mod.compile_logp()(pm_mod.initial_point()))
        """
        self._graph_checks = {
            "add_norm_check": add_norm_check,
            "add_bk_check": add_bk_check,
            "add_solver_success_check": add_solver_success_check,
            "solver_tol": solver_tol,
        }
        super().build_statespace_graph(data=data)

    def _register_additional_statespace_variables(self) -> None:
        add_norm_check = self._graph_checks["add_norm_check"]
        add_bk_check = self._graph_checks["add_bk_check"]
        add_solver_success_check = self._graph_checks["add_solver_success_check"]
        solver_tol = self._graph_checks["solver_tol"]

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
            # These are diagnostics only. Only the Potentials below reject draws.
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
        idata: xr.DataTree | xr.Dataset,
        n_lags: int = 10,
        observed: bool = False,
        lag_step: int = 1,
        compile_kwargs: dict | None = None,
    ) -> xr.DataArray:
        r"""
        Compute the posterior distribution of the model-implied autocorrelation matrices.

        For each posterior draw the stationary state covariance :math:`\Sigma` solves the discrete Lyapunov
        equation :math:`\Sigma = T \Sigma T^\top + R Q R^\top`, and the autocorrelation at lag :math:`k` is
        :math:`T^{k \cdot \texttt{lag\_step}} \Sigma` normalized by the state standard deviations. The whole
        calculation is one PyTensor graph evaluated across every draw at once with
        :func:`pymc.compute_deterministics`.

        Parameters
        ----------
        idata : DataTree or Dataset
            Inference data whose ``posterior`` group holds draws of the model parameters, or that group itself.
        n_lags : int, optional
            Number of non-zero lags to compute. The returned ``lag`` dimension has ``n_lags + 1`` entries. Defaults
            to 10.
        observed : bool, optional
            Return the autocorrelation of the observed series, with measurement error included in the lag-0
            variance, in place of the latent states. Defaults to False.
        lag_step : int, optional
            Spacing between lags, in model periods. For an observable that is a temporal aggregate (for example an
            annual series from a quarterly model), set this to the aggregation period so successive lags are one
            observation apart. Defaults to 1.
        compile_kwargs : dict, optional
            Passed through to :func:`pymc.compute_deterministics`. Defaults to None.

        Returns
        -------
        autocorrelation : DataArray
            Autocorrelation matrices with dimensions ``(chain, draw, lag, state, state_aux)``, where the state
            dimensions are the observed states when ``observed`` is True and the latent states otherwise.

        Examples
        --------
        With ``lag_step=4``, successive lags of a quarterly model are one year apart:

        .. code-block:: python

            import numpy as np
            import pymc as pm
            import xarray as xr

            from gEconpy import statespace_from_gcn
            from gEconpy.data import get_example_gcn

            ss_mod = statespace_from_gcn(get_example_gcn("RBC"), verbose=False)
            ss_mod.configure(observed_states=["Y"], constant_params="auto", verbose=False)
            with pm.Model(coords=ss_mod.coords):
                ss_mod.to_pymc()
                pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)
                ss_mod.build_statespace_graph(np.full((40, 1), np.nan), add_norm_check=False)

            calibration = {**ss_mod.param_dict, "sigma_epsilon_A": 0.01}
            posterior = xr.Dataset(
                {name: (("chain", "draw"), np.full((1, 4), calibration[name])) for name in ss_mod.param_names},
                coords={"chain": [0], "draw": np.arange(4)},
            )
            acf = ss_mod.sample_autocorrelation_matrices(posterior, n_lags=4, lag_step=4)
            print(acf.sel(state="Y", state_aux="Y").mean(["chain", "draw"]).values)
        """
        posterior = idata.posterior if hasattr(idata, "posterior") else idata
        state_dim = OBS_STATE_DIM if observed else ALL_STATE_DIM
        aux_dim = OBS_STATE_AUX_DIM if observed else ALL_STATE_AUX_DIM
        name = "observation_autocorrelation" if observed else "autocorrelation"
        coords = {**self.coords, "lag": np.arange(n_lags + 1)}

        param_dims = {param: list(dims) for param, dims in self.param_dims.items() if dims is not None}

        with pm.Model(coords=coords) as acf_model:
            dummy_graph.build_dummy_graph(self, coords=self.coords, dims=param_dims)
            _, _, _, _, T, Z, R, H, Q = self.unpack_statespace()

            Sigma = pt.linalg.solve_discrete_lyapunov(T, R @ Q @ R.T)
            # Deterministic cumulator and observation-lag copies make the stationary covariance singular.
            if self._n_cumulator_states == 0 and self._n_obs_lag_states == 0:
                Sigma = assume(Sigma, positive_definite=True)

            T_step = pt.linalg.matrix_power(T, lag_step)
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
                autocov_0 = Z @ Sigma @ Z.T + H
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

    def to_pymc(self, exclude_priors: list[str] | None = None) -> None:
        """
        Add the model's parameter and shock priors to the active PyMC model context.

        Parameters
        ----------
        exclude_priors : list of str, optional
            Names of priors to skip. Constant parameters are always skipped. Defaults to none.

        Examples
        --------
        Excluding ``alpha`` leaves room for a hand-written prior under the same name:

        .. code-block:: python

            import pymc as pm

            from gEconpy import statespace_from_gcn
            from gEconpy.data import get_example_gcn

            ss_mod = statespace_from_gcn(get_example_gcn("RBC"), verbose=False)
            ss_mod.configure(observed_states=["Y"], constant_params=["beta", "delta"], verbose=False)

            with pm.Model(coords=ss_mod.coords) as pm_mod:
                ss_mod.to_pymc(exclude_priors=["alpha"])
                pm.Beta("alpha", alpha=20, beta=40)
            print(sorted(rv.name for rv in pm_mod.free_RVs))
        """
        if exclude_priors is None:
            exclude_priors = []

        skip = set(exclude_priors) | set(self.constant_parameters)

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
    statespace_mod: DSGEStateSpace,
    pymc_model: pm.Model,
    index: pd.DatetimeIndex | None = None,
    n_samples: int = 500,
    pct_missing: float = 0,
    random_seed: np.random.Generator | int | None = None,
    mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
    build_statespace_kwargs: dict | None = None,
) -> tuple[xr.Dataset, pd.DataFrame, xr.DataTree]:
    """
    Generate an artificial dataset from one draw of the prior predictive distribution.

    The statespace graph is built into a copy of ``pymc_model``, so the caller's model is left untouched.

    Parameters
    ----------
    statespace_mod : DSGEStateSpace
        Statespace model to generate data from. Must already be configured with :meth:`DSGEStateSpace.configure`.
    pymc_model : Model
        PyMC model holding priors for the DSGE parameters, without a Kalman filter likelihood attached.
    index : DatetimeIndex, optional
        Index of the generated data. Defaults to a quarterly index from 1980-01-01 to 2024-11-01.
    n_samples : int, optional
        Number of prior predictive samples to draw. Defaults to 500.
    pct_missing : float, optional
        Fraction of each column to blank out at random, between 0 and 1. Defaults to 0.
    random_seed : Generator or int, optional
        Seed for every random draw. Defaults to None.
    mvn_method : str, optional
        Multivariate normal sampling method passed to
        :meth:`~pymc_extras.statespace.core.statespace.PyMCStateSpace.sample_unconditional_prior`. Defaults to
        ``"svd"``.
    build_statespace_kwargs : dict, optional
        Keyword arguments forwarded to :meth:`DSGEStateSpace.build_statespace_graph`. Defaults to None.

    Returns
    -------
    true_parameters : Dataset
        Parameter values of the draw that generated the data, plus the draw index as ``param_idx``.
    data : DataFrame
        Generated observations, one column per observed state.
    prior_idata : DataTree
        Prior predictive draws, plus the unconditional prior trajectories under ``unconditional_prior``.

    Examples
    --------
    With ``pct_missing=0.1``, a tenth of every series is blanked out at random:

    .. code-block:: python

        import pymc as pm

        from gEconpy import data_from_prior, statespace_from_gcn
        from gEconpy.data import get_example_gcn

        ss_mod = statespace_from_gcn(get_example_gcn("RBC"), verbose=False)
        ss_mod.configure(observed_states=["Y"], constant_params="auto", verbose=False)

        with pm.Model(coords=ss_mod.coords) as pm_mod:
            ss_mod.to_pymc()
            pm.Gamma("sigma_epsilon_A", alpha=2, beta=100)

        true_params, data, prior_idata = data_from_prior(ss_mod, pm_mod, n_samples=10, pct_missing=0.1, random_seed=0)
        print(float(true_params["sigma_epsilon_A"]), data["Y"].isna().mean())
    """
    rng = np.random.default_rng(random_seed)
    if build_statespace_kwargs is None:
        build_statespace_kwargs = {}

    if index is None:
        index = pd.date_range(start="1980-01-01", end="2024-11-01", freq="QS-OCT")
    dummy_data = pd.DataFrame(np.nan, index=index, columns=statespace_mod.observed_states)
    dummy_data.index.freq = dummy_data.index.inferred_freq

    new_model = pymc_model.copy()
    with new_model:
        statespace_mod.build_statespace_graph(dummy_data, **build_statespace_kwargs)

    with warnings.catch_warnings(action="ignore"), freeze_dims_and_data(new_model):
        prior_idata = pm.sample_prior_predictive(
            n_samples, compile_kwargs={"mode": statespace_mod.mode}, random_seed=rng
        )

    with warnings.catch_warnings(action="ignore"):
        prior_trajectories = statespace_mod.sample_unconditional_prior(
            prior_idata, random_seed=rng, mvn_method=mvn_method
        )

    prior_idata["unconditional_prior"] = prior_trajectories

    draw_idx = rng.choice(prior_idata.prior.coords["draw"].values)

    true_params = prior_idata.prior.isel(chain=0, draw=draw_idx).to_dataset()
    true_params["param_idx"] = draw_idx

    data = prior_trajectories.isel(chain=0, draw=draw_idx).prior_observed
    data = data.to_dataframe().drop(columns=["chain", "draw"]).unstack("observed_state").droplevel(axis=1, level=0)

    data.index.freq = data.index.inferred_freq
    if pct_missing > 0:
        n_missing = int(data.shape[0] * pct_missing)
        for col in data:
            missing_idxs = rng.choice(data.index, size=n_missing, replace=False)
            data.loc[missing_idxs, col] = np.nan

    return true_params, data, prior_idata


def prepare_mixed_frequency_data(
    low_freq_data: pd.DataFrame,
    high_freq: str,
    aggregation_period: int = 4,
    observation_position: Literal["first", "last"] = "last",
) -> pd.DataFrame:
    """
    Expand low-frequency data onto a high-frequency index for mixed-frequency estimation.

    Each low-frequency value lands at the first or last high-frequency period of its aggregation window, with
    ``NaN`` everywhere else. The Kalman filter treats the ``NaN`` entries as missing observations. Flow and stock
    variables are placed identically. The ``temporal_aggregation`` option of :meth:`DSGEStateSpace.configure`
    distinguishes them, where ``"sum"`` makes the observation equation sum over the window.

    Parameters
    ----------
    low_freq_data : DataFrame
        Observed data at low frequency, one column per observed variable, indexed by a ``DatetimeIndex`` at the
        low-frequency periodicity, for example annual.
    high_freq : str
        Pandas frequency string of the model's periodicity, for example "QS" for quarterly.
    aggregation_period : int, optional
        Number of high-frequency periods per low-frequency observation. Defaults to 4, which is annual from
        quarterly.
    observation_position : str, optional
        Whether the low-frequency observation corresponds to the "first" or "last" high-frequency period in each
        window. Defaults to "last".

    Returns
    -------
    high_freq_data : DataFrame
        High-frequency DataFrame with ``NaN`` at unobserved periods.

    Examples
    --------
    The default ``observation_position="last"`` puts each annual value at the fourth quarter of its year:

    .. code-block:: python

        import pandas as pd

        from gEconpy import prepare_mixed_frequency_data

        annual = pd.DataFrame(
            {"GDP": [100.0, 110.0], "R": [0.05, 0.04]},
            index=pd.to_datetime(["2020-01-01", "2021-01-01"]),
        )
        quarterly = prepare_mixed_frequency_data(annual, high_freq="QS")
        print(quarterly)
    """
    pos_idx = 0 if observation_position == "first" else aggregation_period - 1

    first_date = low_freq_data.index.min()
    hf_index = pd.date_range(start=first_date, periods=len(low_freq_data) * aggregation_period, freq=high_freq)

    high_freq_data = pd.DataFrame(np.nan, index=hf_index, columns=list(low_freq_data.columns))

    for lf_date, row in low_freq_data.iterrows():
        window_periods = hf_index[hf_index >= lf_date][:aggregation_period]
        if len(window_periods) <= pos_idx:
            continue
        high_freq_data.loc[window_periods[pos_idx]] = row

    last_obs_idx = high_freq_data.last_valid_index()
    if last_obs_idx is not None:
        high_freq_data = high_freq_data.loc[:last_obs_idx]

    high_freq_data.index.freq = high_freq_data.index.inferred_freq
    return high_freq_data
