class ModelSolve:
    params = [["RBC", "New_Keynesian"], ["cycle_reduction", "gensys"], ["numpy", "pytensor"]]
    param_names = ["model", "solver", "backend"]

    def setup(self, model, solver, backend):
        from gEconpy.data import get_example_gcn
        from gEconpy.model.build import model_from_gcn

        self.model = model_from_gcn(get_example_gcn(model), verbose=False, backend=backend)
        self.model.steady_state(verbose=False)

    def time_solve(self, model, solver, backend):
        self.model.solve_model(solver=solver, verbose=False)

    def peakmem_solve(self, model, solver, backend):
        self.model.solve_model(solver=solver, verbose=False)
