import numpy as np
import pandas as pd
import xarray as xr
from numpy.linalg import LinAlgError

from gEconpy.classes.progress_bar import ProgressBar
from gEconpy.estimation.estimate import build_Q_and_H, build_Z_matrix
from gEconpy.estimation.kalman_filter import kalman_filter
from gEconpy.estimation.kalman_smoother import kalman_smoother


def prior_solvability_check(
    model, n_samples, seed=None, param_subset=None, pert_solver="cycle_reduction"
):
    # Discard the noise priors here, we don't need them
    param_dicts, *_ = model.sample_param_dict_from_prior(n_samples, seed, param_subset)

    data = pd.DataFrame(param_dicts)
    progress_bar = ProgressBar(n_samples, verb="Sampling")

    if pert_solver not in ["cycle_reduction", "gensys"]:
        raise ValueError(
            f'Argument pert_solver must be one of "cycle_reduction" or "gensys", found {pert_solver}'
        )

    def check_solvable(param_dict):
        try:
            results = model.f_ss(param_dict)

            ss_dict = results["ss_dict"]
            calib_dict = results["calib_dict"]
            ss_success = results["success"]
            param_dict = param_dict | calib_dict

        except ValueError:
            return "steady_state"

        if not ss_success:
            return "steady_state"

        try:
            max_iter = 1000
            tol = 1e-18
            verbose = False

            exog, endog = np.array(list(param_dict.values())), np.array(list(ss_dict.values()))
            A, B, C, D = model.build_perturbation_matrices(exog, endog)

            if pert_solver == "cycle_reduction":
                solver = model.perturbation_solver.solve_policy_function_with_cycle_reduction
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
    free_params = model.free_param_dict.copy()
    for param_dict in param_dicts:
        progress_bar.start()
        free_params.update(param_dict)
        result = check_solvable(free_params)
        results.append(result)
        progress_bar.stop()

    data["failure_step"] = results

    return data


def get_initial_time_index(df):
    t0 = df.index[0]

    if isinstance(df.index, pd.DatetimeIndex):
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

        return np.array(t0 - offset, dtype="datetime64")

    else:
        return np.array(t0 - 1)


def simulate_trajectories_from_prior(
    model,
    n_samples=1000,
    n_simulations=100,
    simulation_length=40,
    seed=None,
    param_subset=None,
    pert_kwargs=None,
):

    if pert_kwargs is None:
        pert_kwargs = {}

    simulations = []
    model_var_names = [x.base_name for x in model.variables]
    shock_names = [x.name for x in model.shocks]

    free_param_dicts, shock_dicts, _ = model.sample_param_dict_from_prior(
        n_samples, seed, param_subset
    )
    free_param_dicts = pd.DataFrame(free_param_dicts).T.to_dict()
    shock_dicts = pd.DataFrame(shock_dicts).T.to_dict()

    i = 0
    progress_bar = ProgressBar(n_samples, "Sampling")
    free_params = model.free_param_dict.copy()
    for param_dict, shock_dict in zip(free_param_dicts.values(), shock_dicts.values()):
        progress_bar.start()
        free_params.update(param_dict)

        try:
            model.steady_state(verbose=False)
            model.solve_model(verbose=False, on_failure="ignore", **pert_kwargs)

            data = model.simulate(
                simulation_length=simulation_length,
                n_simulations=n_simulations,
                shock_dict=shock_dict,
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


def safe_get_idx_as_dict(df, idx):
    if idx >= df.shape[0]:
        return {}
    else:
        return df.iloc[idx].to_dict()


def kalman_filter_from_prior(model, data, n_samples, filter_type="univariate", seed=None):
    observed_vars = data.columns.tolist()
    model_var_names = [x.base_name for x in model.variables]
    shock_names = [x.name for x in model.shocks]

    initial_params = model.free_param_dict.copy()

    results = []
    dicts_of_samples = model.sample_param_dict_from_prior(n_samples, seed=seed)
    param_dicts, shock_dicts, noise_dicts = map(pd.DataFrame, dicts_of_samples)

    progress_bar = ProgressBar(n_samples, "Sampling")
    i = 0

    while i < n_samples:
        try:
            param_dict = safe_get_idx_as_dict(param_dicts, i)
            shock_dict = safe_get_idx_as_dict(shock_dicts, i)
            obs_dict = safe_get_idx_as_dict(noise_dicts, i)

            progress_bar.start()
            model.free_param_dict.update(param_dict)

            model.steady_state(verbose=False)
            model.solve_model(verbose=False, on_failure="error")

            T, R = model.T.values, model.R.values
            Z = build_Z_matrix(observed_vars, model_var_names)
            Q, H = build_Q_and_H(shock_dict, shock_names, observed_vars, obs_dict)

            filter_results = kalman_filter(
                data.values, T, Z, R, H, Q, a0=None, P0=None, filter_type=filter_type
            )
            filtered_states, _, filtered_covariances, *_ = filter_results

            smoother_results = kalman_smoother(T, R, Q, filtered_states, filtered_covariances)
            results.append(list(filter_results) + list(smoother_results))

            i += 1
            progress_bar.stop()
        except (ValueError, np.linalg.LinAlgError):
            continue

    coords = {
        "sample": np.arange(n_samples),
        "time": data.index.values,
        "variable": model_var_names,
    }

    pred_coords = {
        "sample": np.arange(n_samples),
        "time": np.r_[
            get_initial_time_index(data),
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
            get_initial_time_index(data),
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
