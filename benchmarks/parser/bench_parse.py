class Parser:
    params = [["RBC", "New_Keynesian"]]
    param_names = ["model"]

    def setup(self, model):
        from gEconpy.data import get_example_gcn
        from gEconpy.parser import parse_gcn

        self.parse_gcn = parse_gcn
        with get_example_gcn(model).open() as f:
            self.gcn_text = f.read()

    def time_parse(self, model):
        self.parse_gcn(self.gcn_text)

    def peakmem_parse(self, model):
        self.parse_gcn(self.gcn_text)
