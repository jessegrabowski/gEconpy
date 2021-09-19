import sympy as sp
import numpy as np
from TimeAwareSympy import TimeAwareSymbol
from warnings import catch_warnings, simplefilter
from scipy import optimize


def rewrite_model_equations_eq_0(model):
    output = []
    for eq in model:
        if isinstance(eq, sp.core.relational.Equality):
            eq = eq.lhs - eq.rhs
        output.append(eq)

    return output


def automatic_try_reduce(model, var_list):
    try_reduce_dict = {}
    for eq in model:
        atoms = eq.atoms()
        if len(atoms) < 4:
            key = np.random.choice(eq.args)
            val = sp.solve(eq, key)[0]
            try_reduce_dict[key] = val

    model = [eq.subs(try_reduce_dict) for eq in model]
    model = [eq for eq in model if eq != 0]

    var_list = [x for x in var_list if x not in try_reduce_dict.keys()]
    return model, var_list


def make_random_x0(var_list, non_random_val_dict=None, low=0.1, high=3):
    x0 = np.random.uniform(low, high, size=len(var_list))

    if non_random_val_dict is not None:
        for x, value in non_random_val_dict.items():
            idx = np.flatnonzero([variable.name == x for variable in var_list])
            if len(idx) == 1:
                x0[idx] = value
    return x0


def convert_model_to_steady_state(model, var_list):
    ss_sub_dict = {}
    for variable in var_list:
        if isinstance(variable, TimeAwareSymbol):
            ss_sub_dict[variable] = variable.to_ss()

    ss_model = []
    for eq in model:
        ss_model.append(eq.subs(ss_sub_dict))
    ss_var_list = list(set(list(ss_sub_dict.values())))

    return ss_model, ss_var_list


def substitute_all_params(model, param_dict):
    model = [eq.subs(param_dict).evalf() for eq in model]
    model = [eq for eq in model if eq != 0]
    return model


def convert_model_to_numpy_function(model, var_list, return_lambdify_array=False, return_jac=True):
    f_model = [sp.lambdify(var_list, eq) for eq in model]

    def f_model_v(x):
        return np.array([eq(*x) for eq in f_model])

    if return_jac:
        f_jac = [[sp.lambdify(var_list, eq.diff(x)) for x in var_list] for eq in model]

        def f_jac_v(x):
            return np.array([[eq(*x) for eq in row] for row in f_jac])

        if return_lambdify_array:
            return f_model_v, f_jac_v, f_model, f_jac
        else:
            return f_model_v, f_jac_v

    if return_lambdify_array:
        return f_model_v, f_model
    else:
        return f_model_v


def compute_ss(f_model,
               ss_var_list,
               computed_ss_values=None,
               n_iter=10000,
               rand_search_range=(0.1, 3),
               f_jac=None,
               xtol=1e-12,
               method='hybr',
               **kwargs):
    for i in range(n_iter):
        print(f'{i:<7} / 10000', end='\r')
        low, high = rand_search_range
        x0 = make_random_x0(ss_var_list, computed_ss_values, low=low, high=high)
        with catch_warnings():
            simplefilter('ignore')
            results = optimize.root(f_model,
                                    jac=f_jac,
                                    x0=x0,
                                    tol=xtol,
                                    method=method,
                                    **kwargs)
            if results.success:
                print('')
                print(f'Done!')
                break

    ss_value_dict = dict(zip(ss_var_list, results.x))
    return results, ss_value_dict
