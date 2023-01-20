import numpy as np
import pandas as pd
import xarray as xr

from gEconpy.classes.progress_bar import ProgressBar
from gEconpy.estimation.estimate import build_Q_and_H, build_Z_matrix, split_param_dict
from gEconpy.estimation.estimation_utilities import split_random_variables
from gEconpy.estimation.kalman_filter import kalman_filter
from gEconpy.estimation.kalman_smoother import kalman_smoother
from gEconpy.sampling.prior_utilities import get_initial_time_index


def simulate_trajectories_from_posterior(
    model, posterior, n_samples=1000, n_simulations=100, simulation_length=40
):
    simulations = []
    model_var_names = [x.base_name for x in model.variables]
    shock_names = [x.base_name for x in model.shocks]

    random_idx = np.random.choice(
        posterior.dims["sample"], replace=False, size=n_samples
    )
    progress_bar = ProgressBar(n_samples, "Sampling")
    for i, idx in enumerate(random_idx):
        param_dict = {
            k: v["data"]
            for k, v in posterior.sel(sample=posterior.sample[idx])
            .to_dict()["data_vars"]
            .items()
        }
        free_param_dict, shock_dict, obs_dict = split_random_variables(
            param_dict, shock_names, model_var_names
        )
        model.free_param_dict.update(free_param_dict)
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

        except ValueError:
            continue

        finally:
            progress_bar.stop()

    simulations = pd.concat(simulations, axis=1)
    return simulations


def kalman_filter_from_posterior(
    model, data, posterior, n_samples=1000, filter_type="univariate"
):
    observed_vars = data.columns.tolist()
    model_var_names = [x.base_name for x in model.variables]
    shock_names = [x.base_name for x in model.shocks]

    results = []
    model_var_names = [x.base_name for x in model.variables]
    shock_names = [x.base_name for x in model.shocks]

    random_idx = np.random.choice(
        posterior.dims["sample"], replace=False, size=n_samples
    )
    progress_bar = ProgressBar(n_samples, "Sampling")
    for idx in random_idx:
        try:
            all_param_dict = {
                k: v["data"]
                for k, v in posterior.sel(sample=posterior.sample[idx])
                .to_dict()["data_vars"]
                .items()
            }

            param_dict, a0_dict, P0_dict = split_param_dict(all_param_dict)
            free_param_dict, shock_dict, noise_dict = split_random_variables(
                param_dict, shock_names, model_var_names
            )

            model.free_param_dict.update(free_param_dict)
            progress_bar.start()

            model.steady_state(verbose=False)
            model.solve_model(verbose=False, on_failure="raise")

            T, R = model.T.values, model.R.values
            T = np.ascontiguousarray(T)
            R = np.ascontiguousarray(R)
            Z = build_Z_matrix(observed_vars, model_var_names)
            Q, H = build_Q_and_H(shock_dict, shock_names, observed_vars, noise_dict)

            a0 = np.array(list(a0_dict.values()))[:, None] if len(a0_dict) > 0 else None
            P0 = (
                np.eye(len(P0_dict)) * np.array(list(P0_dict.keys()))
                if len(P0_dict) > 0
                else None
            )

            filter_results = kalman_filter(
                np.ascontiguousarray(data.values),
                T,
                Z,
                R,
                H,
                Q,
                a0=a0,
                P0=P0,
                filter_type=filter_type,
            )
            filtered_states, _, filtered_covariances, *_ = filter_results

            smoother_results = kalman_smoother(
                T, R, Q, filtered_states, filtered_covariances
            )
            results.append(list(filter_results) + list(smoother_results))

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
