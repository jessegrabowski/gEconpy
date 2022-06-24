from gEcon.classes.model import gEconModel
import pandas as pd
import numpy as np
import dill
import pymc3 as pm
import theano
import theano.tensor as tt

from gEcon_to_sm import DSGE
from statsmodels_ops import Loglike
import multiprocessing
import arviz as az

dill.settings['recurse'] = True
theano.config.floatX = 'float64'


file_path = '../../GCN Files/RBC_steady_state.gcn'
model = gEconModel(file_path, verbose=False)
model.steady_state(verbose=False)
model.solve_model(verbose=False)

initial_params = model.free_param_dict.copy()

data = model.simulate(simulation_length=100, shock_cov_matrix=np.eye(1) * 0.1, n_simulations=1)
data = data.droplevel(axis=1,level=1).T.loc[1:, ['Y']]
data.index = pd.date_range(start='1900-01-01', freq='AS-JAN', periods=99)

mod = DSGE(model,
           data,
           initialization='stationary')

loglike = Loglike(mod)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    with pm.Model():
        alpha = pm.TruncatedNormal("alpha", mu=0.3, sigma=0.1, lower=0.1, upper=0.8)
        sigma_epsilon_A = pm.InverseGamma('sigma_epsilon_A', mu=0.1, sigma=0.3)

        theta = tt.as_tensor_variable([alpha, sigma_epsilon_A])

        pm.DensityDist("likelihood", loglike, observed=theta)

        trace = pm.sample(return_inferencedata=True, cores=6, target_accept=0.95, pickle_backend='dill')

    az.plot_trace(trace)
