import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError

from gEcon.classes.progress_bar import ProgressBar


def prior_solvability_check(model, n_samples, seed=None, param_subset=None, pert_solver='cycle_reduction'):
    data = pd.DataFrame(model.sample_param_dict_from_prior(n_samples, seed, param_subset))
    progress_bar = ProgressBar(n_samples, verb='Sampling')

    def check_solvable(param_dict):
        try:
            ss_dict, calib_dict = model.f_ss(param_dict)
            resids = model.f_ss_resid(**ss_dict, **calib_dict, **param_dict)
            ss_success = (np.array(resids) ** 2).sum() < 1e-8
        except ValueError:
            return 'steady_state'

        if not ss_success:
            return 'steady_state'

        try:
            max_iter = 1000
            tol = 1e-18
            verbose = False

            A, B, C, D = model.build_perturbation_matrices(**param_dict, **ss_dict)
            if pert_solver == 'cycle_reduction':
                solver = model.perturbation_solver.solve_policy_function_with_cycle_reduction
                T, R, result, log_norm = solver(A, B, C, D, max_iter, tol, verbose)
                pert_success = log_norm < 1e-8

            elif pert_solver == 'gensys':
                solver = model.perturbation_solver.solve_policy_function_with_gensys
                G_1, constant, impact, f_mat, f_wt, y_wt, gev, eu, loose = solver(A, B, C, D, tol, verbose)
                T = G_1[:model.n_variables, :][:, :model.n_variables]
                R = impact[:model.n_variables, :]
                pert_success = G_1 is not None

            else:
                raise NotImplementedError

        except (ValueError, LinAlgError):
            return 'perturbation'

        if not pert_success:
            return 'perturbation'

        bk_success = model.check_bk_condition(system_matrices=[A, B, C, D], verbose=False, return_value='bool')
        if not bk_success:
            return 'blanchard-kahn'

        _, variables, _ = model.perturbation_solver.make_all_variable_time_combinations()
        gEcon_matrices = model.perturbation_solver.statespace_to_gEcon_representation(A, T, R, variables, tol)
        P, Q, _, _, A_prime, R_prime, S_prime = gEcon_matrices

        resid_norms = model.perturbation_solver.residual_norms(B, C, D, Q, P, A_prime, R_prime, S_prime)
        norm_deterministic, norm_stochastic = resid_norms

        if norm_deterministic > 1e-8:
            return 'deterministic_norm'
        if norm_stochastic > 1e-8:
            return 'stochastic_norm'

        return None

    param_dicts = data.T.to_dict().values()
    results = []

    # TODO: How to parallelize this? The problem is the huge model object causes massive overhead.
    for param_dict in param_dicts:
        progress_bar.start()
        result = check_solvable(param_dict)
        results.append(result)
        progress_bar.stop()

    data['failure_step'] = results

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
