# from gEconpy import (
#     classes,
#     # estimation,
#     parser,
#     plotting,
#     sampling,
#     shared,
#     solvers,
# )
#
# # from gEconpy.classes import gEconModel
# from gEconpy.shared import compile_to_statsmodels, make_mod_file
#
__version__ = "1.2.1"
# __all__ = [
#     # "gEconModel",
#     "classes",
#     "estimation",
#     "exceptions",
#     "parser",
#     "plotting",
#     "sampling",
#     "shared",
#     "solvers",
#     "make_mod_file",
#     "compile_to_statsmodels",
# ]

import logging

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)
