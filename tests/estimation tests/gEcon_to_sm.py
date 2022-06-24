import numdifftools as ndt
import statsmodels.api as sm
from transformers import RangeTransformer, PositiveTransformer
import numpy as np


import numdifftools as ndt
import statsmodels.api as sm
from transformers import RangeTransformer, PositiveTransformer
import numpy as np


class DSGE(sm.tsa.statespace.MLEModel):

    def __init__(self, model, data, initialization, params_to_estimate, start_param_values, transform_dict, x0=None, P0=None):
        k_states = model.n_variables
        k_observed = data.shape[1]
        k_posdef = model.n_shocks
        observed = data.columns

        self.model = model
        self.shock_names = [x.base_name for x in self.model.shocks]
        self.dsge_params = [x for x in list(model.free_param_dict.keys()) if 'sigma_epsilon' not in x]
                
        self.params_to_estimate = params_to_estimate
        self.start_param_values = start_param_values

        super(DSGE, self).__init__(endog=data, k_states=k_states, k_posdef=k_posdef,
                                   initialization=initialization,
                                   constant=x0,
                                   initial_state_cov=P0)

        self.ssm['design'][np.arange(k_observed),
                           np.argwhere(np.array([x.base_name in data.columns for x in model.variables])).ravel()] = 1
        self.ssm['state_cov'] = np.eye(k_posdef) * 0.1

        self.state_cov_idxs = np.arange(16)[[any([shock_name in name for name in self.param_names])
                                             for shock_name in self.shock_names]]

        self.transform_dict = transform_dict

    @property
    def param_names(self):
        return self.params_to_estimate

    @property
    def state_names(self):
        return [x.base_name for x in self.model.variables]

    @property
    def start_params(self):
        params = self.start_param_values
        return params

    def transform_params(self, real_line_params):
        '''
        Take in optimizer values on R and map them into parameter space.

        Example: variances must be positive, so apply x ** 2.
        '''
        param_space_params = np.zeros_like(real_line_params)
        for i, (name, param) in enumerate(zip(self.param_names, real_line_params)):
            param_space_params[i] = self.transform_dict[name].real_line_to_param_space(param)

        return param_space_params

    def untransform_params(self, param_space_params):
        '''
        Take in parameters living in the parameter space and apply an "inverse transform"
        to put them back to where the optimizer's last guess was.

        Example: We applied x ** 2 to ensure x is positive, apply x ** (1 / 2).
        '''

        real_line_params = np.zeros_like(param_space_params)
        for i, (name, param) in enumerate(zip(self.param_names, param_space_params)):
            real_line_params[i] = self.transform_dict[name].real_line_to_param_space(param)

        return real_line_params

    def make_param_update_dict(self, params):
        shock_names = self.shock_names
        state_names = self.state_names
        dsge_params = self.dsge_params

        param_names = self.param_names

        param_update_dict = {}
        shock_params = []
        observation_noise_params = []

        for name, param in zip(param_names, params):
            if name in dsge_params:
                param_update_dict[name] = param

            elif any([shock_name in name for shock_name in shock_names]):
                shock_params.append(param)

            else:
                observation_noise_params.append(param)

        return param_update_dict, np.array(shock_params), np.array(observation_noise_params)

    def update(self, params, **kwargs):
        params = super(DSGE, self).update(params, **kwargs)

        update_dict, shock_params, _ = self.make_param_update_dict(params)
        self.model.free_param_dict.update(update_dict)
        
        try:
            self.model.steady_state(verbose=False)
            self.model.solve_model(verbose=False)
        except Exception as e:
            print('np.array([' + ','.join([f'{x:0.3f}' for x in params]) + '])') 
            raise e
#         condition_satisfied = model.check_bk_condition(verbose=False, return_value='bool')
    
#         if not condition_satisfied:
#             raise RuntimeError('The Blanchard-Khan condition was not satisifed, model does NOT have a unique solution.')

        self.ssm['transition'] = self.model.T.values
        self.ssm['selection'] = self.model.R.values

        cov_idx = self.state_cov_idxs
        self.ssm['state_cov', cov_idx, cov_idx] = shock_params

    def score(self, params):
        jac = ndt.Jacobian(self.loglike, step=1e-4, method='central')
        return jac(params)[-1]

    def hessian(self, params):
        Hfun = ndt.Hessian(self.loglike, step=1e-4, method='central')
        return Hfun(params)[-1]
