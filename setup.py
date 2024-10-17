import versioneer

from setuptools import find_packages, setup
from setuptools.dist import Distribution

dist = Distribution()
dist.parse_config_files()


NAME: str = dist.get_name()  # type: ignore


if __name__ == "__main__":
    setup(
        name=NAME,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        packages=find_packages(exclude=["tests*"]),
    )
