class PerfectForesight:
    params = [["RBC"], [50, 100, 200], ["numpy", "pytensor"]]
    param_names = ["model", "simulation_length", "backend"]

    def setup(self, model, simulation_length, backend):
        from gEconpy.data import get_example_gcn
        from gEconpy.model.build import model_from_gcn
        from gEconpy.model.perfect_foresight import solve_perfect_foresight

        self.model = model_from_gcn(get_example_gcn(model), verbose=False, backend=backend)
        self.ss = self.model.steady_state(verbose=False)
        self.solve_perfect_foresight = solve_perfect_foresight

        # Use first state variable for initial condition shock
        first_var = self.model.variables[0].base_name
        ss_val = self.ss[f"{first_var}_ss"]
        self.initial_conditions = {first_var: ss_val * 0.9}

    def time_solve(self, model, simulation_length, backend):
        self.solve_perfect_foresight(
            self.model,
            simulation_length=simulation_length,
            initial_conditions=self.initial_conditions,
        )

    def peakmem_solve(self, model, simulation_length, backend):
        self.solve_perfect_foresight(
            self.model,
            simulation_length=simulation_length,
            initial_conditions=self.initial_conditions,
        )
