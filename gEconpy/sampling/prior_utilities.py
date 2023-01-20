import numpy as np
import pandas as pd
import xarray as xr
from numpy.linalg import LinAlgError

from gEconpy.classes.progress_bar import ProgressBar
from gEconpy.estimation.estimate import build_Q_and_H, build_Z_matrix
from gEconpy.estimation.estimation_utilities import split_random_variables
from gEconpy.estimation.kalman_filter import kalman_filter
from gEconpy.estimation.kalman_smoother import kalman_smoother


def prior_solvability_check(
    model, n_samples, seed=None, param_subset=None, pert_solver="cycle_reduction"
):
    data = pd.DataFrame(
        model.sample_param_dict_from_prior(
            n_samples, seed, param_subset, sample_shock_sigma=True
        )
    )
    progress_bar = ProgressBar(n_samples, verb="Sampling")

    def check_solvable(param_dict):
        try:
            ss_dict, calib_dict = model.f_ss(param_dict)
            resids = model.f_ss_resid(**ss_dict, **calib_dict, **param_dict)
            ss_success = (np.array(resids) ** 2).sum() < 1e-8
        except ValueError:
            return "steady_state"

        if not ss_success:
            return "steady_state"

        try:
            max_iter = 1000
            tol = 1e-18
            verbose = False

            A, B, C, D = model.build_perturbation_matrices(**param_dict, **ss_dict)
            if pert_solver == "cycle_reduction":
                solver = (
                    model.perturbation_solver.solve_policy_function_with_cycle_reduction
                )
                T, R, result, log_norm = solver(A, B, C, D, max_iter, tol, verbose)
                pert_success = log_norm < 1e-8

            elif pert_solver == "gensys":
                solver = model.perturbation_solver.solve_policy_function_with_gensys
                G_1, constant, impact, f_mat, f_wt, y_wt, gev, eu, loose = solver(
                    A, B, C, D, tol, verbose
                )
                T = G_1[: model.n_variables, :][:, : model.n_variables]
                R = impact[: model.n_variables, :]
                pert_success = G_1 is not None

            else:
                raise NotImplementedError

        except (ValueError, LinAlgError):
            return "perturbation"

        if not pert_success:
            return "perturbation"

        bk_success = model.check_bk_condition(
            system_matrices=[A, B, C, D], verbose=False, return_value="bool"
        )
        if not bk_success:
            return "blanchard-kahn"

        (
            _,
            variables,
            _,
        ) = model.perturbation_solver.make_all_variable_time_combinations()
        gEcon_matrices = model.perturbation_solver.statespace_to_gEcon_representation(
            A, T, R, variables, tol
        )
        P, Q, _, _, A_prime, R_prime, S_prime = gEcon_matrices

        resid_norms = model.perturbation_solver.residual_norms(
            B, C, D, Q, P, A_prime, R_prime, S_prime
        )
        norm_deterministic, norm_stochastic = resid_norms

        if norm_deterministic > 1e-8:
            return "deterministic_norm"
        if norm_stochastic > 1e-8:
            return "stochastic_norm"

        return None

    param_dicts = data.T.to_dict().values()
    results = []

    # TODO: How to parallelize this? The problem is the huge model object causes massive overhead.
    for param_dict in param_dicts:
        progress_bar.start()
        result = check_solvable(param_dict)
        results.append(result)
        progress_bar.stop()

    data["failure_step"] = results

    return data


def get_initial_time_index(df):
    t0 = df.index[0]
    freq = df.index.inferred_freq
    base_freq = freq.split("-")[0]

    if "Q" in base_freq:
        offset = pd.DateOffset(months=3)
    elif "M" in base_freq:
        offset = pd.DateOffset(months=1)
    elif "A" in base_freq:
        offset = pd.DateOffset(years=1)
    else:
        raise NotImplementedError("Data isn't one of: Quarterly, Monthly, Annual")

    return t0 - offset


def simulate_trajectories_from_prior(
    model, n_samples=1000, n_simulations=100, simulation_length=40
):
    simulations = []
    model_var_names = [x.base_name for x in model.variables]
    shock_names = [x.name for x in model.shocks]

    param_dicts = pd.DataFrame(
        model.sample_param_dict_from_prior(n_samples)
    ).T.to_dict()
    i = 0

    progress_bar = ProgressBar(n_samples, "Sampling")
    for param_dict in param_dicts.values():
        # free_param_dict, shock_dict, obs_dict = split_random_variables(param_dict, shock_names, model_var_names)
        model.free_param_dict.update(param_dict)
        progress_bar.start()

        try:
            model.steady_state(verbose=False)
            model.solve_model(verbose=False, on_failure="ignore")

            data = model.simulate(
                simulation_length=simulation_length,
                n_simulations=n_simulations,
                show_progress_bar=False,
            )
            simulaton_ids = np.arange(n_simulations).astype(int)

            data = data.rename(
                axis=1,
                level=1,
                mapper=dict(zip(simulaton_ids, simulaton_ids + (n_simulations * i))),
            )

            simulations.append(data)
            i += 1

        except ValueError:
            continue

        finally:
            progress_bar.stop()

    simulations = pd.concat(simulations, axis=1)
    return simulations


def kalman_filter_from_prior(model, data, n_samples, filter_type="univariate"):
    observed_vars = data.columns.tolist()
    model_var_names = [x.base_name for x in model.variables]
    shock_names = [x.base_name for x in model.shocks]

    results = []
    param_dicts = pd.DataFrame(
        model.sample_param_dict_from_prior(n_samples, sample_shock_sigma=True)
    ).T.to_dict()

    progress_bar = ProgressBar(n_samples, "Sampling")
    i = 0

    while i < n_samples:
        try:
            param_dict = param_dicts[i]
            param_dict, shock_dict, obs_dict = split_random_variables(
                param_dict, shock_names, observed_vars
            )
            model.free_param_dict.update(param_dict)

            progress_bar.start()
            model.steady_state(verbose=False)
            model.solve_model(verbose=False, on_failure="raise")

            T, R = model.T.values, model.R.values
            Z = build_Z_matrix(observed_vars, model_var_names)
            Q, H = build_Q_and_H(shock_dict, shock_names, observed_vars, obs_dict)

            filter_results = kalman_filter(
                data.values, T, Z, R, H, Q, a0=None, P0=None, filter_type=filter_type
            )
            filtered_states, _, filtered_covariances, *_ = filter_results

            smoother_results = kalman_smoother(
                T, R, Q, filtered_states, filtered_covariances
            )
            results.append(list(filter_results) + list(smoother_results))

            i += 1
            progress_bar.stop()
        except ValueError:
            continue

    coords = {
        "sample": np.arange(n_samples),
        "time": data.index.values,
        "variable": model_var_names,
    }

    pred_coords = {
        "sample": np.arange(n_samples),
        "time": np.r_[
            np.array(get_initial_time_index(data), dtype="datetime64"),
            data.index.values,
        ],
        "variable": model_var_names,
    }

    cov_coords = {
        "sample": np.arange(n_samples),
        "time": data.index.values,
        "variable": model_var_names,
        "variable2": model_var_names,
    }

    pred_cov_coords = {
        "sample": np.arange(n_samples),
        "time": np.r_[
            np.array(get_initial_time_index(data), dtype="datetime64"),
            data.index.values,
        ],
        "variable": model_var_names,
        "variable2": model_var_names,
    }

    kf_data = xr.Dataset(
        {
            "Filtered_State": xr.DataArray(
                data=np.stack([results[i][0] for i in range(n_samples)]), coords=coords
            ),
            "Predicted_State": xr.DataArray(
                data=np.stack([results[i][1] for i in range(n_samples)]),
                coords=pred_coords,
            ),
            "Smoothed_State": xr.DataArray(
                data=np.stack([results[i][5] for i in range(n_samples)]), coords=coords
            ),
            "Filtered_Cov": xr.DataArray(
                data=np.stack([results[i][2] for i in range(n_samples)]),
                coords=cov_coords,
            ),
            "Predicted_Cov": xr.DataArray(
                data=np.stack([results[i][3] for i in range(n_samples)]),
                coords=pred_cov_coords,
            ),
            "Smoothed_Cov": xr.DataArray(
                data=np.stack([results[i][6] for i in range(n_samples)]),
                coords=cov_coords,
            ),
            "loglikelihood": xr.DataArray(
                data=np.stack([results[i][4] for i in range(n_samples)]),
                coords={"sample": np.arange(n_samples), "time": data.index.values},
            ),
        }
    )

    return kf_data
