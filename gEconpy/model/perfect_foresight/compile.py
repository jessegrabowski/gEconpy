from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pytensor
import pytensor.tensor as pt
import sympy as sp

from pytensor.graph.replace import graph_replace
from sympytensor import as_tensor

from gEconpy.classes.containers import SymbolDictionary
from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import build_symbolic_jacobian, make_cache_key
from gEconpy.model.model import Model
from gEconpy.model.perfect_foresight.assemble import (
    StackedJacobianLayout,
    build_stacked_jacobian_layout,
)
from gEconpy.model.timing import classify_variables_by_timing
from gEconpy.utilities import safe_to_ss


@dataclass
class PerfectForesightProblem:
    """
    Compiled single-period functions of a model, plus the names that fix their input and output ordering.

    Attributes
    ----------
    f_resid_and_jac : callable
        Function returning the residuals and the ``(n_eq, 3 * n_vars)`` Jacobian of one period's equations. It is
        called as ``f(y_tm1, y_t, y_tp1, [x_t,] *params)``, with ``x_t`` omitted when the model has no shocks.
    f_resid_only : callable or None
        Function with the same inputs returning only the residuals, or None when it was not compiled.
    jacobian_sparsity : ndarray of bool
        Structural nonzero mask of the single-period Jacobian, of shape ``(n_eq, 3 * n_vars)``. It is derived
        from the equations, so it holds at every point the solver can visit.
    var_names : list of str
        Variable names, giving the column order of the stacked system.
    shock_names : list of str
        Shock names, giving the order of ``x_t``.
    param_names : list of str
        Parameter names, in the order the compiled functions expect them.
    T : int
        Number of periods in the simulation horizon.
    """

    f_resid_and_jac: Callable
    f_resid_only: Callable | None
    jacobian_sparsity: np.ndarray
    var_names: list[str]
    shock_names: list[str]
    param_names: list[str]
    T: int

    @property
    def n_vars(self) -> int:
        """Number of model variables."""
        return len(self.var_names)

    @property
    def n_shocks(self) -> int:
        """Number of model shocks."""
        return len(self.shock_names)

    @property
    def n_eq(self) -> int:
        """Number of model equations, which equals the number of variables."""
        return len(self.var_names)

    @cached_property
    def jacobian_layout(self) -> StackedJacobianLayout:
        """Index arrays of the stacked Jacobian, resolved once and reused across Newton iterations."""
        return build_stacked_jacobian_layout(self.jacobian_sparsity, self.n_vars, self.n_eq, self.T)


def compile_perfect_foresight_problem(
    model: Model,
    T: int,
    **compile_kwargs,
) -> PerfectForesightProblem:
    """
    Compile the single-period residual and Jacobian functions used by the perfect foresight solver.

    Parameters
    ----------
    model : Model
        Model to compile. Every steady-state variable that appears in its equations must have an analytic solution.
    T : int
        Number of periods in the simulation horizon.
    **compile_kwargs
        Keyword arguments forwarded to :func:`pytensor.function`. ``on_unused_input`` defaults to ``"ignore"``.

    Returns
    -------
    problem : PerfectForesightProblem
        The compiled functions, together with the variable, shock, and parameter names.
    """
    graph = _build_single_period_graph(model)
    compile_kwargs.setdefault("on_unused_input", "ignore")

    f_resid_and_jac = pytensor.function(graph.inputs, [graph.residuals, graph.jacobian], **compile_kwargs)
    f_resid_only = pytensor.function(graph.inputs, [graph.residuals], **compile_kwargs)

    return PerfectForesightProblem(
        f_resid_and_jac=f_resid_and_jac,
        f_resid_only=f_resid_only,
        jacobian_sparsity=graph.sparsity,
        var_names=graph.var_names,
        shock_names=graph.shock_names,
        param_names=graph.param_names,
        T=T,
    )


@dataclass
class _SinglePeriodGraph:
    inputs: list[pt.TensorVariable]
    residuals: pt.TensorVariable
    jacobian: pt.TensorVariable
    sparsity: np.ndarray
    var_names: list[str]
    shock_names: list[str]
    param_names: list[str]


def _build_single_period_graph(model: Model) -> _SinglePeriodGraph:
    """
    Build the symbolic residual and Jacobian of one period's equations as functions of stacked input vectors.

    The model equations are converted to pytensor once, differentiated with respect to every variable at ``t-1``,
    ``t``, and ``t+1``, and then rewritten so that each scalar variable reads from an element of ``y_tm1``, ``y_t``,
    ``y_tp1``, or ``x_t``. A variable that never appears at some time index gets a structurally zero Jacobian column
    there.
    """
    shock_names = [s.base_name for s in model.shocks]

    equations = _substitute_steady_state_values(model.equations, model._ss_solution_dict)
    vars_tm1, vars_t, vars_tp1, shocks_t = classify_variables_by_timing(equations, shock_names)

    cache: dict = {}
    equations_pt = [as_tensor(eq, cache=cache) for eq in equations]

    def get_pt_var(sym: TimeAwareSymbol) -> pt.TensorVariable:
        return cache[make_cache_key(sym.name, cls=TimeAwareSymbol)]

    params_in_equations = []
    params_pt = []
    for param in model.params + model.deterministic_params:
        key = make_cache_key(param.name, cls=sp.Symbol)
        if key in cache:
            params_in_equations.append(param)
            params_pt.append(cache[key])

    var_names = [x.base_name for x in vars_t]
    vars_t_pt = [get_pt_var(x) for x in vars_t]
    shocks_pt = [get_pt_var(x) for x in shocks_t]
    n_vars = len(vars_t_pt)
    n_shocks = len(shocks_pt)

    # Variables absent at t-1 or t+1 get a dummy scalar so every time index has one node per variable.
    tm1_by_name = {x.base_name: get_pt_var(x) for x in vars_tm1}
    tp1_by_name = {x.base_name: get_pt_var(x) for x in vars_tp1}
    vars_tm1_pt = [tm1_by_name.get(name, pt.dscalar(f"{name}_tm1_dummy")) for name in var_names]
    vars_tp1_pt = [tp1_by_name.get(name, pt.dscalar(f"{name}_tp1_dummy")) for name in var_names]

    # The shared cache makes the Jacobian's column nodes the same objects as the vars_*_pt lists wherever the
    # variable appears at that time index, so the vector replacement below lines up.
    jacobian_wrt = [v.set_t(-1) for v in vars_t] + list(vars_t) + [v.set_t(1) for v in vars_t]
    jacobian = build_symbolic_jacobian(equations, jacobian_wrt, cache, to_ss=False)
    sparsity = _jacobian_sparsity(equations, jacobian_wrt)
    residuals = pt.stack(equations_pt)

    y_tm1 = pt.dvector("y_tm1", shape=(n_vars,))
    y_t = pt.dvector("y_t", shape=(n_vars,))
    y_tp1 = pt.dvector("y_tp1", shape=(n_vars,))
    x_t = pt.dvector("x_t", shape=(n_shocks,)) if n_shocks > 0 else None

    replacements = {}
    for i in range(n_vars):
        replacements[vars_tm1_pt[i]] = y_tm1[i]
        replacements[vars_t_pt[i]] = y_t[i]
        replacements[vars_tp1_pt[i]] = y_tp1[i]
    if x_t is not None:
        for i, shock in enumerate(shocks_pt):
            replacements[shock] = x_t[i]

    residuals_vec, jacobian_vec = graph_replace([residuals, jacobian], replacements, strict=False)

    inputs = [y_tm1, y_t, y_tp1]
    if x_t is not None:
        inputs.append(x_t)
    inputs.extend(params_pt)

    return _SinglePeriodGraph(
        inputs=inputs,
        residuals=residuals_vec,
        jacobian=jacobian_vec,
        sparsity=sparsity,
        var_names=var_names,
        shock_names=[x.base_name for x in shocks_t],
        param_names=[p.name for p in params_in_equations],
    )


def _jacobian_sparsity(equations: list[sp.Expr], wrt: list[TimeAwareSymbol]) -> np.ndarray:
    """
    Structural nonzero mask of ``d(equations)/d(wrt)``.

    Parameters
    ----------
    equations : list of sympy expression
        Equations to differentiate, one per row of the mask.
    wrt : list of TimeAwareSymbol
        Symbols to differentiate with respect to, one per column of the mask.

    Returns
    -------
    sparsity : ndarray of bool
        Mask of shape ``(len(equations), len(wrt))``, True wherever the derivative can be nonzero.
    """
    # Membership rather than differentiation: a symbol absent from an equation has an identically zero
    # derivative there, so the mask is a superset of the true nonzero set at a fraction of the cost. The
    # superset direction is the safe one, since a missing entry would silently corrupt the Newton step.
    symbols_by_equation = [equation.free_symbols for equation in equations]

    return np.array([[symbol in symbols for symbol in wrt] for symbols in symbols_by_equation], dtype=bool)


def _substitute_steady_state_values(
    equations: list[sp.Expr],
    ss_solution_dict: SymbolDictionary | None,
) -> list[sp.Expr]:
    """
    Replace every steady-state variable in the equations with its analytic expression.

    Parameters
    ----------
    equations : list of sympy.Expr
        Model equations that may contain ``X_ss`` symbols.
    ss_solution_dict : SymbolDictionary, optional
        Analytically known steady-state solutions, keyed by steady-state variable.

    Returns
    -------
    equations : list of sympy.Expr
        Equations with no steady-state variables left.
    """
    ss_atoms = {a for eq in equations for a in eq.atoms(TimeAwareSymbol) if a.time_index == "ss"}
    if not ss_atoms:
        return equations

    values_by_name = {}
    if ss_solution_dict:
        values_by_name = {safe_to_ss(key).name: value for key, value in ss_solution_dict.to_sympy().items()}
    sub_dict = {atom: values_by_name[atom.name] for atom in ss_atoms if atom.name in values_by_name}

    remaining = ss_atoms - set(sub_dict.keys())
    if remaining:
        names = ", ".join(sorted(str(a) for a in remaining))
        raise ValueError(
            f"Perfect foresight simulation requires all steady-state variables to have analytic "
            f"solutions, but the following do not: {names}. Provide analytic steady-state values "
            f"in the STEADY_STATE block of your GCN file."
        )

    return [eq.subs(sub_dict) for eq in equations]
