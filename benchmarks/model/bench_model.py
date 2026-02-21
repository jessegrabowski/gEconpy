class ModelFunctions:
    params = [["RBC", "New_Keynesian"], ["numpy", "pytensor"]]
    param_names = ["model", "backend"]

    def setup(self, model, backend):
        from gEconpy import (
            autocovariance_matrix,
            check_bk_condition,
            impulse_response_function,
            simulate,
            stationary_covariance_matrix,
        )
        from gEconpy.data import get_example_gcn
        from gEconpy.model.build import model_from_gcn

        self.model = model_from_gcn(get_example_gcn(model), verbose=False, backend=backend)
        self.model.steady_state(verbose=False)
        self.T, self.R = self.model.solve_model(verbose=False)

        self.stationary_covariance_matrix = stationary_covariance_matrix
        self.autocovariance_matrix = autocovariance_matrix
        self.check_bk_condition = check_bk_condition
        self.impulse_response_function = impulse_response_function
        self.simulate = simulate

    def time_stationary_covariance_matrix(self, model, backend):
        self.stationary_covariance_matrix(self.model, T=self.T, R=self.R, shock_std=1.0)

    def time_autocovariance_matrix(self, model, backend):
        self.autocovariance_matrix(self.model, T=self.T, R=self.R, shock_std=1.0, n_lags=10)

    def time_check_bk_condition(self, model, backend):
        self.check_bk_condition(self.model, verbose=False)

    def time_impulse_response_function(self, model, backend):
        self.impulse_response_function(self.model, T=self.T, R=self.R, simulation_length=40, shock_size=1.0)

    def time_simulate(self, model, backend):
        self.simulate(self.model, T=self.T, R=self.R, simulation_length=100, shock_std=1.0)

    def peakmem_stationary_covariance_matrix(self, model, backend):
        self.stationary_covariance_matrix(self.model, T=self.T, R=self.R, shock_std=1.0)

    def peakmem_autocovariance_matrix(self, model, backend):
        self.autocovariance_matrix(self.model, T=self.T, R=self.R, shock_std=1.0, n_lags=10)

    def peakmem_simulate(self, model, backend):
        self.simulate(self.model, T=self.T, R=self.R, simulation_length=100, shock_std=1.0)
