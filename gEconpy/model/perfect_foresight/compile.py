from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytensor
import pytensor.tensor as pt
import sympy as sp

from pytensor.graph.replace import graph_replace
from sympytensor import as_tensor

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import make_cache_key
from gEconpy.model.timing import classify_variables_by_timing
from gEconpy.pytensorf.sparse_jacobian import sparse_jacobian
from gEconpy.utilities import safe_to_ss

if TYPE_CHECKING:
    from gEconpy.model.model import Model


def _substitute_steady_state_values(
    equations: list[sp.Expr],
    ss_solution_dict,
) -> list[sp.Expr]:
    """Replace steady-state variables in equations with their analytic expressions.

    Parameters
    ----------
    equations : list of sp.Expr
        Model equations that may contain ``X_ss`` symbols.
    ss_solution_dict : SymbolDictionary
        Analytically known steady-state solutions mapping ``X_ss`` to expressions.

    Returns
    -------
    equations : list of sp.Expr
        Equations with steady-state variables substituted away.

    Raises
    ------
    ValueError
        If any steady-state variables remain that lack analytic solutions.
    """
    ss_atoms = {a for eq in equations for a in eq.atoms(TimeAwareSymbol) if a.time_index == "ss"}
    if not ss_atoms:
        return equations

    sub_dict = {}
    if ss_solution_dict:
        sympy_dict = ss_solution_dict.to_sympy()
        for atom in ss_atoms:
            for key, value in sympy_dict.items():
                if safe_to_ss(key).name == atom.name:
                    sub_dict[atom] = value
                    break

    remaining = ss_atoms - set(sub_dict.keys())
    if remaining:
        names = ", ".join(sorted(str(a) for a in remaining))
        raise ValueError(
            f"Perfect foresight simulation requires all steady-state variables to have analytic "
            f"solutions, but the following do not: {names}. Provide analytic steady-state values "
            f"in the STEADY_STATE block of your GCN file."
        )

    return [eq.subs(sub_dict) for eq in equations]


def _build_jacobian_var_lists(
    var_names: list[str],
    tm1_by_name: dict[str, pt.TensorVariable],
    tp1_by_name: dict[str, pt.TensorVariable],
    vars_t_pt: list[pt.TensorVariable],
) -> tuple[list[pt.TensorVariable], list[pt.TensorVariable], list[pt.TensorVariable]]:
    """Build variable lists for all time indices, creating dummies where needed."""
    jac_vars_tm1 = [tm1_by_name.get(name, pt.dscalar(f"{name}_tm1_dummy")) for name in var_names]
    jac_vars_t = list(vars_t_pt)
    jac_vars_tp1 = [tp1_by_name.get(name, pt.dscalar(f"{name}_tp1_dummy")) for name in var_names]
    return jac_vars_tm1, jac_vars_t, jac_vars_tp1


def _build_vector_replacements(
    var_names: list[str],
    jac_vars_tm1: list[pt.TensorVariable],
    jac_vars_t: list[pt.TensorVariable],
    jac_vars_tp1: list[pt.TensorVariable],
    shocks_pt: list[pt.TensorVariable],
    y_tm1_vec: pt.TensorVariable,
    y_t_vec: pt.TensorVariable,
    y_tp1_vec: pt.TensorVariable,
    x_t_vec: pt.TensorVariable | None,
) -> dict:
    """Map scalar variables to vector element indexing."""
    replacements = {}
    for i in range(len(var_names)):
        replacements[jac_vars_tm1[i]] = y_tm1_vec[i]
        replacements[jac_vars_t[i]] = y_t_vec[i]
        replacements[jac_vars_tp1[i]] = y_tp1_vec[i]
    if x_t_vec is not None:
        for i, var in enumerate(shocks_pt):
            replacements[var] = x_t_vec[i]
    return replacements


def _compile_single_period_function(
    model: "Model",
    **compile_kwargs,
) -> tuple[Callable, list[str], list[str], list[str]]:
    """Compile single-period residual and Jacobian function.

    Returns
    -------
    f : Callable
        Pytensor function computing (residuals, jacobian) for one period.
    var_names : list of str
        Variable names in column order.
    shock_names : list of str
        Shock names in column order.
    param_names : list of str
        Parameter names in input order.
    """
    shock_names = [s.base_name for s in model.shocks]

    equations = _substitute_steady_state_values(model.equations, model._ss_solution_dict)
    vars_tm1, vars_t, vars_tp1, shocks_t = classify_variables_by_timing(equations, shock_names)

    cache: dict = {}
    equations_pt = [as_tensor(eq, cache=cache) for eq in equations]

    def get_pt_var(sym: TimeAwareSymbol) -> pt.TensorVariable:
        return cache[make_cache_key(sym.name, cls=TimeAwareSymbol)]

    # Collect parameters that actually appear in the equations (both free and deterministic)
    all_param_symbols = model.params + model.deterministic_params
    params_in_equations = []
    params_pt = []
    for p in all_param_symbols:
        key = make_cache_key(p.name, cls=sp.Symbol)
        if key in cache:
            params_in_equations.append(p)
            params_pt.append(cache[key])
    param_names = [p.name for p in params_in_equations]

    vars_t_pt = [get_pt_var(x) for x in vars_t]
    shocks_pt = [get_pt_var(x) for x in shocks_t]

    n_vars = len(vars_t_pt)
    n_shocks = len(shocks_pt)
    var_names = [x.base_name for x in vars_t]

    tm1_by_name = {x.base_name: get_pt_var(x) for x in vars_tm1}
    tp1_by_name = {x.base_name: get_pt_var(x) for x in vars_tp1}

    jac_vars_tm1, jac_vars_t, jac_vars_tp1 = _build_jacobian_var_lists(var_names, tm1_by_name, tp1_by_name, vars_t_pt)

    jac_vars = jac_vars_tm1 + jac_vars_t + jac_vars_tp1
    jacobian = sparse_jacobian(equations_pt, jac_vars, return_sparse=False)
    residuals = pt.stack(equations_pt)

    y_tm1_vec = pt.dvector("y_tm1", shape=(n_vars,))
    y_t_vec = pt.dvector("y_t", shape=(n_vars,))
    y_tp1_vec = pt.dvector("y_tp1", shape=(n_vars,))
    x_t_vec = pt.dvector("x_t", shape=(n_shocks,)) if n_shocks > 0 else None

    vec_replacements = _build_vector_replacements(
        var_names, jac_vars_tm1, jac_vars_t, jac_vars_tp1, shocks_pt, y_tm1_vec, y_t_vec, y_tp1_vec, x_t_vec
    )

    residuals_vec = graph_replace(residuals, vec_replacements, strict=False)
    jacobian_vec = graph_replace(jacobian, vec_replacements, strict=False)

    func_inputs = [y_tm1_vec, y_t_vec, y_tp1_vec]
    if x_t_vec is not None:
        func_inputs.append(x_t_vec)
    func_inputs.extend(params_pt)

    if "on_unused_input" not in compile_kwargs:
        compile_kwargs["on_unused_input"] = "ignore"

    f = pytensor.function(func_inputs, [residuals_vec, jacobian_vec], **compile_kwargs)

    return f, var_names, [x.base_name for x in shocks_t], param_names


def _compile_single_period_residual_function(
    model: "Model",
    **compile_kwargs,
) -> Callable:
    """Compile single-period residual-only function (no Jacobian).

    This is a cheap evaluation function used during line search to avoid computing the
    Jacobian at rejected trial points.

    Returns
    -------
    f : Callable
        Pytensor function computing only residuals for one period.
    """
    shock_names = [s.base_name for s in model.shocks]

    equations = _substitute_steady_state_values(model.equations, model._ss_solution_dict)
    vars_tm1, vars_t, vars_tp1, shocks_t = classify_variables_by_timing(equations, shock_names)

    cache: dict = {}
    equations_pt = [as_tensor(eq, cache=cache) for eq in equations]

    def get_pt_var(sym: TimeAwareSymbol) -> pt.TensorVariable:
        return cache[make_cache_key(sym.name, cls=TimeAwareSymbol)]

    all_param_symbols = model.params + model.deterministic_params
    params_pt = []
    for p in all_param_symbols:
        key = make_cache_key(p.name, cls=sp.Symbol)
        if key in cache:
            params_pt.append(cache[key])

    vars_t_pt = [get_pt_var(x) for x in vars_t]
    shocks_pt_list = [get_pt_var(x) for x in shocks_t]

    n_vars = len(vars_t_pt)
    n_shocks = len(shocks_pt_list)
    var_names = [x.base_name for x in vars_t]

    tm1_by_name = {x.base_name: get_pt_var(x) for x in vars_tm1}
    tp1_by_name = {x.base_name: get_pt_var(x) for x in vars_tp1}

    jac_vars_tm1, jac_vars_t, jac_vars_tp1 = _build_jacobian_var_lists(var_names, tm1_by_name, tp1_by_name, vars_t_pt)

    residuals = pt.stack(equations_pt)

    y_tm1_vec = pt.dvector("y_tm1", shape=(n_vars,))
    y_t_vec = pt.dvector("y_t", shape=(n_vars,))
    y_tp1_vec = pt.dvector("y_tp1", shape=(n_vars,))
    x_t_vec = pt.dvector("x_t", shape=(n_shocks,)) if n_shocks > 0 else None

    vec_replacements = _build_vector_replacements(
        var_names, jac_vars_tm1, jac_vars_t, jac_vars_tp1, shocks_pt_list, y_tm1_vec, y_t_vec, y_tp1_vec, x_t_vec
    )

    residuals_vec = graph_replace(residuals, vec_replacements, strict=False)

    func_inputs = [y_tm1_vec, y_t_vec, y_tp1_vec]
    if x_t_vec is not None:
        func_inputs.append(x_t_vec)
    func_inputs.extend(params_pt)

    if "on_unused_input" not in compile_kwargs:
        compile_kwargs["on_unused_input"] = "ignore"

    return pytensor.function(func_inputs, [residuals_vec], **compile_kwargs)


@dataclass
class PerfectForesightProblem:
    """Compiled perfect foresight problem."""

    f_resid_and_jac: Callable
    f_resid_only: Callable | None
    var_names: list[str]
    shock_names: list[str]
    param_names: list[str]
    T: int

    @property
    def n_vars(self) -> int:
        return len(self.var_names)

    @property
    def n_shocks(self) -> int:
        return len(self.shock_names)

    @property
    def n_eq(self) -> int:
        return len(self.var_names)


def compile_perfect_foresight_problem(
    model: "Model",
    T: int,
    **compile_kwargs,
) -> PerfectForesightProblem:
    """Compile the single-period dynamic function for perfect foresight simulation."""
    f_resid_and_jac, var_names, shock_names, param_names = _compile_single_period_function(model, **compile_kwargs)
    f_resid_only = _compile_single_period_residual_function(model, **compile_kwargs)
    return PerfectForesightProblem(
        f_resid_and_jac=f_resid_and_jac,
        f_resid_only=f_resid_only,
        var_names=var_names,
        shock_names=shock_names,
        param_names=param_names,
        T=T,
    )
