class SteadyState:
    params = [["RBC", "New_Keynesian"], ["numpy", "pytensor"]]
    param_names = ["model", "backend"]

    def setup(self, model, backend):
        from gEconpy.data import get_example_gcn
        from gEconpy.model.build import model_from_gcn

        self.model = model_from_gcn(get_example_gcn(model), verbose=False, backend=backend)

    def time_steady_state(self, model, backend):
        self.model.steady_state(verbose=False)

    def peakmem_steady_state(self, model, backend):
        self.model.steady_state(verbose=False)
