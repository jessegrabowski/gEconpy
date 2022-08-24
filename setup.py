from setuptools import setup

setup(
    entry_points={
        "numba_extensions": [
            "init = gEcon:init",
        ]
    })