from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING

import pytensor
import pytensor.tensor as pt
import sympy as sp

from pytensor.graph.replace import graph_replace
from sympytensor import as_tensor

from gEconpy.classes.time_aware_symbol import TimeAwareSymbol
from gEconpy.model.compile import make_cache_key
from gEconpy.pytensorf.sparse_jacobian import sparse_jacobian

if TYPE_CHECKING:
    from gEconpy.model.model import Model


def _collect_time_aware_atoms(equations: list[sp.Expr]) -> set[TimeAwareSymbol]:
    """Collect all TimeAwareSymbol atoms from a list of equations."""
    return reduce(lambda a, b: a.union(b), (eq.atoms(TimeAwareSymbol) for eq in equations), set())


def _classify_variables_by_timing(
    equations: list[sp.Expr],
    shock_names: list[str],
) -> tuple[list[TimeAwareSymbol], list[TimeAwareSymbol], list[TimeAwareSymbol], list[TimeAwareSymbol]]:
    """Classify variables by time index into (vars_tm1, vars_t, vars_tp1, shocks_t)."""
    all_atoms = _collect_time_aware_atoms(equations)
    shock_name_set = set(shock_names)

    def is_shock(x: TimeAwareSymbol) -> bool:
        return x.base_name in shock_name_set

    def vars_at_time(t: int) -> list[TimeAwareSymbol]:
        return sorted(
            [x for x in all_atoms if x.time_index == t and not is_shock(x)],
            key=lambda x: x.base_name,
        )

    shocks_t = sorted([x for x in all_atoms if is_shock(x)], key=lambda x: x.base_name)
    return vars_at_time(-1), vars_at_time(0), vars_at_time(1), shocks_t


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
    param_names = [p.name for p in model.params]

    vars_tm1, vars_t, vars_tp1, shocks_t = _classify_variables_by_timing(model.equations, shock_names)

    cache: dict = {}
    equations_pt = [as_tensor(eq, cache=cache) for eq in model.equations]

    def get_pt_var(sym: TimeAwareSymbol) -> pt.TensorVariable:
        return cache[make_cache_key(sym.name, cls=TimeAwareSymbol)]

    def get_pt_param(sym: sp.Symbol) -> pt.TensorVariable:
        return cache[make_cache_key(sym.name, cls=sp.Symbol)]

    vars_t_pt = [get_pt_var(x) for x in vars_t]
    shocks_pt = [get_pt_var(x) for x in shocks_t]
    params_pt = [get_pt_param(p) for p in model.params]

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


@dataclass
class PerfectForesightProblem:
    """Compiled perfect foresight problem."""

    f_resid_and_jac: Callable
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
    return PerfectForesightProblem(
        f_resid_and_jac=f_resid_and_jac,
        var_names=var_names,
        shock_names=shock_names,
        param_names=param_names,
        T=T,
    )
