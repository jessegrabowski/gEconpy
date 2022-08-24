from setuptools import setup

setup(
    ...,
    entry_points={
        "numba_extensions": [
            "init = numba_linalg:init",
        ]
    },
    ...
)