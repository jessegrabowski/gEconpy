import pandas as pd
import numpy as np
from gEcon.classes.progress_bar import ProgressBar


def prior_steady_state_check(model, n_samples, seed=None):
    data = pd.DataFrame(model.sample_param_dict_from_prior(n_samples, seed))
    progress_bar = ProgressBar(n_samples, verb='Sampling')

    def check_steady_state_solvable(param_dict):
        try:
            model.f_ss(param_dict)
            return True
        except ValueError:
            return False

    param_dicts = data.T.to_dict().values()
    results = []

    # TODO: How to parallelize this? The problem is the huge model object causes massive overhead.
    for param_dict in param_dicts:
        progress_bar.start()
        result = check_steady_state_solvable(param_dict)
        results.append(result)
        progress_bar.stop()

    data['success'] = results

    return data


def simulate_trajectories_from_prior(model, n_samples=1000, n_simulations=100, simulation_length=40):
    simulations = []
    param_dicts = pd.DataFrame(model.sample_param_dict_from_prior(n_samples)).T.to_dict()
    i = 0

    progress_bar = ProgressBar(n_samples, 'Sampling')
    for param_dict in param_dicts.values():
        model.free_param_dict = param_dict
        progress_bar.start()
        try:
            model.steady_state(verbose=False)
            model.solve_model(verbose=False, on_failure='ignore')

            data = model.simulate(simulation_length=simulation_length, n_simulations=n_simulations,
                                  show_progress_bar=False)
            simulaton_ids = np.arange(n_simulations).astype(int)

            data = data.rename(axis=1, level=1, mapper=dict(zip(simulaton_ids, simulaton_ids + (n_simulations * i))))

            simulations.append(data)
            i += 1

        except ValueError:
            continue

        finally:
            progress_bar.stop()

    simulations = pd.concat(simulations, axis=1)
    return simulations
