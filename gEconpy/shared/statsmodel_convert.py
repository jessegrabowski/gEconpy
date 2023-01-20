from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.kalman_filter import INVERT_UNIVARIATE, SOLVE_LU
from statsmodels.tsa.statespace.mlemodel import MLEModel, _handle_args

from gEconpy.classes.transformers import IdentityTransformer, PositiveTransformer


def compile_to_statsmodels(model):
    """
    Compile a gEconModel object into a Statsmodels MLEModel object.

    Statsmodels includes a full suite of tools for solving and fitting linear state space
    models via Maximum Likelihood. This function takes a solved gEconpy model object
    and uses it to implement a `statsmodels.tsa.statespace` state space model.

    Parameters
    ----------
    model : gEconModel
        A gEconModel object to be compiled into a Statsmodels MLEModel object.

    Returns
    -------
    MLEModel
        A Statsmodels MLEModel object compiled from the gEconModel object.

    """

    class DSGEModel(MLEModel):
        def __init__(
            self,
            data: pd.DataFrame,
            initialization: str,
            param_start_dict: Dict[str, float],
            shock_start_dict: Dict[str, float],
            noise_start_dict: Optional[Dict[str, float]] = None,
            param_transforms: Optional[Dict[str, Callable]] = None,
            shock_transforms: Optional[Dict[str, Callable]] = None,
            noise_transforms: Optional[Dict[str, Callable]] = None,
            x0: Optional[np.ndarray] = None,
            P0: Optional[np.ndarray] = None,
            fit_MAP: bool = False,
            **kwargs,
        ):
            """
            Create a DSGEModel object for maximum-likelihood estimation, subclassed from
            `statsmodels.tsa.statespace.MLEModel`.

            Parameters
            ----------
            model: A DSGE model object
                The model object to be used to create the DSGEModel
            data: pd.DataFrame
                A pandas DataFrame containing the data to be used for estimation
            initialization: string
                The type of Kalman filter initialization to use.  One of 'approximate_diffuse',
                'stationary', 'known', 'fixed', 'diffuse' or 'none'
            param_start_dict: dict
                A dictionary of parameter starting values, where keys are parameter names
                and values are floats. Parameters not included this dictionary will not be
                estimated when `.fit()`. is called.
            shock_start_dict: dict
                A dictionary of shock variance starting values, where keys are shock names and values
                are floats. All shocks not include in this dictionary will be dropped from the model
                when `.fit()` is called.
            noise_start_dict: dict, optional
                A dictionary of observation noise starting values, where keys are observed state names
                and values are floats. Default is zero for all observed variables.
            param_transforms: dict, optional
                A dictionary of functions to transform parameters before they are passed to the likelihood
                function.  Keys are parameter names, values are functions. Default is the identity
                function for all parameters.
            shock_transforms: dict, optional
                A dictionary of functions to transform shock variance terms before they are passed to
                the likelihood function.  Keys are shock names, values are functions. Default is
                the square function for all variances.
            noise_transforms: dict, optional
                A dictionary of functions to transform observation noise variances before they are
                 passed to the likelihood function.  Keys are noise state names, values are
                  functions. Default is the square function for all variannces.
            x0: array_like, optional
                A 1-d array of starting values for the state vector
            P0: array_like, optional
                A 2-d array of starting values for the state covariance matrix
            fit_MAP: bool, optional
                If True, fit the model in maximum a posteriori (MAP) sense rather than maximum
                likelihood sense.  Defaults to False.
            kwargs:
                Additional arguments to pass to the MLEModel constructor
            """
            k_states = model.n_variables
            k_observed = data.shape[1]
            k_posdef = model.n_shocks

            noise_start_dict = noise_start_dict or {}

            self.model = model
            self.data = data
            self.fit_MAP = fit_MAP

            self.shock_names = [x.base_name for x in self.model.shocks]
            self.dsge_params = list(model.free_param_dict.keys())

            param_priors = self.model.param_priors.copy()
            shock_priors = self.model.shock_priors.copy()
            noise_priors = self.model.observation_noise_priors.copy()

            self.prior_dict = param_priors.copy()
            self.prior_dict.update(
                {k: d.rv_params["scale"] for k, d in shock_priors.items()}
            )
            self.prior_dict.update(noise_priors)

            n_shocks = len(self.shock_names)

            self.params_to_estimate = list(param_start_dict.keys())
            self.shocks_to_estimate = list(shock_start_dict.keys())
            self.noisy_states = list(noise_start_dict.keys())

            self.start_dict = param_start_dict.copy()
            self.start_dict.update(shock_start_dict)
            self.start_dict.update(noise_start_dict)

            self._validate_start_dict(
                param_start_dict, shock_start_dict, noise_start_dict
            )
            self._build_transform_dict(
                param_transforms, shock_transforms, noise_transforms
            )
            self._validate_priors(param_priors, shock_priors, noise_priors)

            super().__init__(
                endog=data,
                k_states=k_states,
                k_posdef=k_posdef,
                initialization=initialization,
                constant=x0,
                initial_state_cov=P0,
                **kwargs,
            )

            model_names = [x.base_name for x in model.variables]
            missing_vars = [x for x in data.columns if x not in model_names]
            if any(missing_vars):
                msg = "Data contains the following columns not associated with variables in the model:"
                msg += ", ".join(missing_vars)
                raise ValueError(msg)

            Z_idx = [model_names.index(x) for x in data.columns if x in model_names]

            self.ssm["design"][np.arange(k_observed), Z_idx] = 1
            self.ssm["state_cov"] = np.eye(k_posdef) * 0.1
            self.ssm["obs_cov"] = np.zeros((k_observed, k_observed))

            self.state_cov_idxs = (
                np.arange(n_shocks, dtype="int"),
                np.arange(n_shocks, dtype="int"),
            )
            self.obs_cov_idxs = np.where(np.isin(data.columns, self.noisy_states))

        def _validate_start_dict(
            self,
            param_start_dict: Dict[str, float],
            shock_start_dict: Dict[str, float],
            noise_start_dict: Dict[str, float],
        ) -> None:
            """
            Validate that all the parameters, shocks, and observation noises that are supposed to be
             estimated have starting values, and that any starting values provided correspond to
             parameters, shocks, or observation noises that exist in the model or data.

            Parameters
            ----------
            param_start_dict: Dict[str, float]
                A dictionary of starting values for parameters that are to be estimated.
            shock_start_dict: Dict[str, float]
                A dictionary of starting values for shocks that are to be estimated.
            noise_start_dict: Dict[str, float]
                A dictionary of starting values for observation noises that are to be estimated.
            """
            missing_vars = [
                x for x in self.params_to_estimate if x not in param_start_dict.keys()
            ]
            missing_shocks = [
                x for x in self.shocks_to_estimate if x not in shock_start_dict.keys()
            ]
            missing_noise = [
                x for x in self.noisy_states if x not in noise_start_dict.keys()
            ]
            msg = (
                "The following {} to be estimated were not assigned a starting value: "
            )

            if any(missing_vars):
                raise ValueError(msg.format("parameters") + ", ".join(missing_vars))

            if any(missing_shocks):
                raise ValueError(msg.format("shocks") + ", ".join(missing_shocks))

            if any(missing_noise):
                raise ValueError(
                    msg.format("observation noises") + ", ".join(missing_noise)
                )

            extra_vars = [
                x
                for x in param_start_dict.keys()
                if x not in self.model.free_param_dict.keys()
            ]
            extra_shocks = [
                x
                for x in shock_start_dict.keys()
                if x not in [x.base_name for x in self.model.shocks]
            ]
            extra_noise = [
                x for x in noise_start_dict.keys() if x not in self.data.columns
            ]

            msg = "The following {} were given starting values, but did not appear in the {}: "
            if any(extra_vars):
                raise ValueError(
                    msg.format("parameters", "model definition") + ", ".join(extra_vars)
                )

            if any(extra_shocks):
                raise ValueError(
                    msg.format("shocks", "model definition") + ", ".join(missing_shocks)
                )

            if any(extra_noise):
                raise ValueError(
                    msg.format("observation noises", "data") + ", ".join(missing_noise)
                )

        def _build_transform_dict(
            self, param_transforms, shock_transforms, noise_transforms
        ):
            self.transform_dict = {}
            for param in self.params_to_estimate:
                if param in param_transforms.keys():
                    self.transform_dict[param] = param_transforms[param]
                else:
                    print(
                        f"Parameter {param} was not assigned a transformation, assigning IdentityTransform"
                    )
                    self.transform_dict[param] = IdentityTransformer()

            if shock_transforms is None:
                self.transform_dict.update(
                    {k: PositiveTransformer() for k in self.shocks_to_estimate}
                )
            else:
                for shock in self.shocks_to_estimate:
                    if shock in shock_transforms.keys():
                        self.transform_dict[shock] = shock_transforms[shock]
                    else:
                        print(
                            f"Shock {shock} was not assigned a transformation, assigning IdentityTransform"
                        )
                        self.transform_dict[shock] = IdentityTransformer()

            if noise_transforms is None:
                self.transform_dict.update(
                    {k: PositiveTransformer() for k in self.noisy_states}
                )
            else:
                for noise in self.noisy_states:
                    if noise in noise_transforms.keys():
                        self.transform_dict[noise] = noise_transforms[noise]
                    else:
                        print(
                            f"Noise for state {noise} was not assigned a transformation, assigning IdentityTransform"
                        )
                        self.transform_dict[noise] = IdentityTransformer()

        def _validate_priors(self, param_priors, shock_priors, noise_priors):
            if not self.fit_MAP:
                return

            missing_vars = [
                x for x in self.params_to_estimate if x not in param_priors.keys()
            ]
            missing_shocks = [
                x for x in self.shocks_to_estimate if x not in shock_priors.keys()
            ]
            missing_noise = [
                x for x in self.noisy_states if x not in noise_priors.keys()
            ]
            msg = "The following {} to be estimated were not assigned a prior: "
            if any(missing_vars):
                raise ValueError(msg.format("parameters") + ", ".join(missing_vars))

            if any(missing_shocks):
                raise ValueError(msg.format("shocks") + ", ".join(missing_shocks))

            if any(missing_noise):
                raise ValueError(
                    msg.format("observation noises") + ", ".join(missing_noise)
                )

        @property
        def param_names(self):
            shock_names = [f"sigma2.{x}" for x in self.shocks_to_estimate]
            noise_names = [f"sigma2.{x}" for x in self.noisy_states]
            return self.params_to_estimate + shock_names + noise_names

        @property
        def external_param_names(self):
            return self.params_to_estimate + self.shocks_to_estimate + self.noisy_states

        @property
        def state_names(self):
            return [x.base_name for x in self.model.variables]

        @property
        def start_params(self):
            param_names = self.external_param_names
            start_params = []

            for name in param_names:
                start_params.append(self.start_dict[name])
            return np.array(start_params)

        def unpack_statespace(self):
            T = np.ascontiguousarray(self.ssm["transition"])
            Z = np.ascontiguousarray(self.ssm["design"])
            R = np.ascontiguousarray(self.ssm["selection"])
            H = np.ascontiguousarray(self.ssm["obs_cov"])
            Q = np.ascontiguousarray(self.ssm["state_cov"])

            return T, Z, R, H, Q

        def transform_params(self, real_line_params):
            """
            Take in optimizer values on R and map them into parameter space.

            Example: variances must be positive, so apply x ** 2.
            """
            param_space_params = np.zeros_like(real_line_params)
            for i, (name, param) in enumerate(
                zip(self.external_param_names, real_line_params)
            ):
                param_space_params[i] = self.transform_dict[name].constrain(param)

            return param_space_params

        def untransform_params(self, param_space_params):
            """
            Take in parameters living in the parameter space and apply an "inverse transform"
            to put them back to where the optimizer's last guess was.

            Example: We applied x ** 2 to ensure x is positive, apply x ** (1 / 2).
            """
            real_line_params = np.zeros_like(param_space_params)
            for i, (name, param) in enumerate(
                zip(self.external_param_names, param_space_params)
            ):
                real_line_params[i] = self.transform_dict[name].unconstrain(param)

            return real_line_params

        def make_param_update_dict(self, params):
            shock_names = self.shock_names
            dsge_params = self.dsge_params
            param_names = self.external_param_names

            param_update_dict = {}
            shock_params = []
            observation_noise_params = []

            for name, param in zip(param_names, params):
                if name in dsge_params:
                    param_update_dict[name] = param
                elif name in shock_names:
                    shock_params.append(param)
                else:
                    observation_noise_params.append(param)

            return (
                param_update_dict,
                np.array(shock_params),
                np.array(observation_noise_params),
            )

        def update(self, params, **kwargs):
            params = super().update(params, **kwargs)

            update_dict, shock_params, obs_params = self.make_param_update_dict(params)
            # original_params = model.free_param_dict.copy()

            self.model.free_param_dict.update(update_dict)
            try:
                self.model.steady_state(verbose=False)
                self.model.solve_model(verbose=False)
                pert_success = True
            except np.linalg.LinAlgError:
                pert_success = False

            condition_satisfied = model.check_bk_condition(
                verbose=False, return_value="bool"
            )

            self.ssm["transition"] = self.model.T.values
            self.ssm["selection"] = self.model.R.values

            cov_idx = self.state_cov_idxs
            self.ssm["state_cov", cov_idx, cov_idx] = shock_params

            obs_idx = self.obs_cov_idxs
            self.ssm["obs_cov", obs_idx, obs_idx] = obs_params

            return pert_success & condition_satisfied

        def loglike(self, params, *args, **kwargs):
            """
            Loglikelihood evaluation

            Parameters
            ----------
            params : array_like
                Array of parameters at which to evaluate the loglikelihood
                function.
            transformed : bool, optional
                Whether or not `params` is already transformed. Default is True.
            **kwargs
                Additional keyword arguments to pass to the Kalman filter. See
                `KalmanFilter.filter` for more details.

            See Also
            --------
            update : modifies the internal state of the state space model to
                     reflect new params

            Notes
            -----
            [1]_ recommend maximizing the average likelihood to avoid scale issues;
            this is done automatically by the base Model fit method.

            References
            ----------
            .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
               Statistical Algorithms for Models in State Space Using SsfPack 2.2.
               Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.
            """
            transformed, includes_fixed, complex_step, kwargs = _handle_args(
                MLEModel._loglike_param_names,
                MLEModel._loglike_param_defaults,
                *args,
                **kwargs,
            )

            params = self.handle_params(
                params, transformed=transformed, includes_fixed=includes_fixed
            )
            success = self.update(
                params,
                transformed=transformed,
                includes_fixed=includes_fixed,
                complex_step=complex_step,
            )

            if complex_step:
                kwargs["inversion_method"] = INVERT_UNIVARIATE | SOLVE_LU

            if success:
                loglike = self.ssm.loglike(complex_step=complex_step, **kwargs)
                if self.fit_MAP:
                    for name, param in zip(self.external_param_names, params):
                        loglike += max(-1e6, self.prior_dict[name].logpdf(param))

            else:
                # If the parameters are invalid, return a large negative number
                loglike = -1e6

            # Koopman, Shephard, and Doornik recommend maximizing the average
            # likelihood to avoid scale issues, but the averaging is done
            # automatically in the base model `fit` method
            return loglike

        def loglikeobs(
            self,
            params,
            transformed=True,
            includes_fixed=False,
            complex_step=False,
            **kwargs,
        ):
            """
            Loglikelihood evaluation

            Parameters
            ----------
            params : array_like
                Array of parameters at which to evaluate the loglikelihood
                function.
            transformed : bool, optional
                Whether or not `params` is already transformed. Default is True.
            **kwargs
                Additional keyword arguments to pass to the Kalman filter. See
                `KalmanFilter.filter` for more details.

            See Also
            --------
            update : modifies the internal state of the Model to reflect new params

            Notes
            -----
            [1]_ recommend maximizing the average likelihood to avoid scale issues;
            this is done automatically by the base Model fit method.

            References
            ----------
            .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
               Statistical Algorithms for Models in State Space Using SsfPack 2.2.
               Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.
            """
            params = self.handle_params(
                params, transformed=transformed, includes_fixed=includes_fixed
            )

            # If we're using complex-step differentiation, then we cannot use
            # Cholesky factorization
            if complex_step:
                kwargs["inversion_method"] = INVERT_UNIVARIATE | SOLVE_LU

            success = self.update(
                params,
                transformed=transformed,
                includes_fixed=includes_fixed,
                complex_step=complex_step,
            )

            if success:
                ll_obs = self.ssm.loglikeobs(complex_step=complex_step, **kwargs)
                if self.fit_MAP:
                    for name, param in zip(self.external_param_names, params):
                        ll_obs += (
                            max(-1e6, self.prior_dict[name].logpdf(param)) / self.nobs
                        )
                return ll_obs

            else:
                # Large negative likelihood for all observations if the parameters are invalid
                return np.full(self.endog.shape[0], -1e6)

        def fit(
            self,
            start_params=None,
            transformed=True,
            includes_fixed=False,
            cov_type=None,
            cov_kwds=None,
            method="lbfgs",
            maxiter=50,
            full_output=1,
            disp=5,
            callback=None,
            return_params=False,
            optim_score=None,
            optim_complex_step=None,
            optim_hessian=None,
            flags=None,
            low_memory=False,
            **kwargs,
        ):
            """
            Fits the model by maximum likelihood via Kalman filter.

            Parameters
            ----------
            start_params : array_like, optional
                Initial guess of the solution for the loglikelihood maximization.
                If None, the default is given by Model.start_params.
            transformed : bool, optional
                Whether or not `start_params` is already transformed. Default is
                True.
            includes_fixed : bool, optional
                If parameters were previously fixed with the `fix_params` method,
                this argument describes whether or not `start_params` also includes
                the fixed parameters, in addition to the free parameters. Default
                is False.
            cov_type : str, optional
                The `cov_type` keyword governs the method for calculating the
                covariance matrix of parameter estimates. Can be one of:

                - 'opg' for the outer product of gradient estimator
                - 'oim' for the observed information matrix estimator, calculated
                  using the method of Harvey (1989)
                - 'approx' for the observed information matrix estimator,
                  calculated using a numerical approximation of the Hessian matrix.
                - 'robust' for an approximate (quasi-maximum likelihood) covariance
                  matrix that may be valid even in the presence of some
                  misspecifications. Intermediate calculations use the 'oim'
                  method.
                - 'robust_approx' is the same as 'robust' except that the
                  intermediate calculations use the 'approx' method.
                - 'none' for no covariance matrix calculation.

                Default is 'opg' unless memory conservation is used to avoid
                computing the loglikelihood values for each observation, in which
                case the default is 'approx'.
            cov_kwds : dict or None, optional
                A dictionary of arguments affecting covariance matrix computation.

                **opg, oim, approx, robust, robust_approx**

                - 'approx_complex_step' : bool, optional - If True, numerical
                  approximations are computed using complex-step methods. If False,
                  numerical approximations are computed using finite difference
                  methods. Default is True.
                - 'approx_centered' : bool, optional - If True, numerical
                  approximations computed using finite difference methods use a
                  centered approximation. Default is False.
            method : str, optional
                The `method` determines which solver from `scipy.optimize`
                is used, and it can be chosen from among the following strings:

                - 'newton' for Newton-Raphson
                - 'nm' for Nelder-Mead
                - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
                - 'lbfgs' for limited-memory BFGS with optional box constraints
                - 'powell' for modified Powell's method
                - 'cg' for conjugate gradient
                - 'ncg' for Newton-conjugate gradient
                - 'basinhopping' for global basin-hopping solver

                The explicit arguments in `fit` are passed to the solver,
                with the exception of the basin-hopping solver. Each
                solver has several optional arguments that are not the same across
                solvers. See the notes section below (or scipy.optimize) for the
                available arguments and for the list of explicit arguments that the
                basin-hopping solver supports.
            maxiter : int, optional
                The maximum number of iterations to perform.
            full_output : bool, optional
                Set to True to have all available output in the Results object's
                mle_retvals attribute. The output is dependent on the solver.
                See LikelihoodModelResults notes section for more information.
            disp : bool, optional
                Set to True to print convergence messages.
            callback : callable callback(xk), optional
                Called after each iteration, as callback(xk), where xk is the
                current parameter vector.
            return_params : bool, optional
                Whether or not to return only the array of maximizing parameters.
                Default is False.
            optim_score : {'harvey', 'approx'} or None, optional
                The method by which the score vector is calculated. 'harvey' uses
                the method from Harvey (1989), 'approx' uses either finite
                difference or complex step differentiation depending upon the
                value of `optim_complex_step`, and None uses the built-in gradient
                approximation of the optimizer. Default is None. This keyword is
                only relevant if the optimization method uses the score.
            optim_complex_step : bool, optional
                Whether or not to use complex step differentiation when
                approximating the score; if False, finite difference approximation
                is used. Default is True. This keyword is only relevant if
                `optim_score` is set to 'harvey' or 'approx'.
            optim_hessian : {'opg','oim','approx'}, optional
                The method by which the Hessian is numerically approximated. 'opg'
                uses outer product of gradients, 'oim' uses the information
                matrix formula from Harvey (1989), and 'approx' uses numerical
                approximation. This keyword is only relevant if the
                optimization method uses the Hessian matrix.
            low_memory : bool, optional
                If set to True, techniques are applied to substantially reduce
                memory usage. If used, some features of the results object will
                not be available (including smoothed results and in-sample
                prediction), although out-of-sample forecasting is possible.
                Default is False.
            **kwargs
                Additional keyword arguments to pass to the optimizer.

            Returns
            -------
            results
                Results object holding results from fitting a state space model.

            See Also
            --------
            statsmodels.base.model.LikelihoodModel.fit
            statsmodels.tsa.statespace.mlemodel.MLEResults
            statsmodels.tsa.statespace.structural.UnobservedComponentsResults
            """

            # Disable complex step approximations by default
            optim_complex_step = optim_complex_step or False
            cov_kwds = cov_kwds or {
                "approx_complex_step": False,
                "approx_centered": True,
            }

            return super().fit(
                start_params=start_params,
                transformed=transformed,
                includes_fixed=includes_fixed,
                cov_type=cov_type,
                cov_kwds=cov_kwds,
                method=method,
                maxiter=maxiter,
                full_output=full_output,
                disp=disp,
                callback=callback,
                return_params=return_params,
                optim_score=optim_score,
                optim_complex_step=optim_complex_step,
                optim_hessian=optim_hessian,
                flags=flags,
                low_memory=low_memory,
                **kwargs,
            )

    return DSGEModel
