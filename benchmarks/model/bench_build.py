class ModelBuild:
    params = [
        ["RBC", "RBC_two_household", "New_Keynesian", "skilled_unskilled_rbc"],
        ["numpy", "pytensor"],
    ]
    param_names = ["model", "backend"]

    def setup(self, model, backend):
        from gEconpy.data import get_example_gcn
        from gEconpy.model.build import model_from_gcn

        self.model_from_gcn = model_from_gcn
        self.gcn_path = get_example_gcn(model)

    def time_build(self, model, backend):
        self.model_from_gcn(self.gcn_path, verbose=False, backend=backend)

    def peakmem_build(self, model, backend):
        self.model_from_gcn(self.gcn_path, verbose=False, backend=backend)
