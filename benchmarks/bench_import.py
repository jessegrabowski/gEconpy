import sys


class Import:
    def _clear_imports(self):
        for k in [k for k in sys.modules if k.startswith("gEconpy")]:
            del sys.modules[k]

    def setup(self):
        self._clear_imports()

    def teardown(self):
        self._clear_imports()

    def time_import_gEconpy(self):
        import gEconpy  # noqa: F401

    def time_import_model_build(self):
        from gEconpy.model import build  # noqa: F401

    def time_import_parser(self):
        from gEconpy import parser  # noqa: F401
