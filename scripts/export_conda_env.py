import re
import subprocess

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "conda_envs" / "environment_dev.yml"

# Pinned so the file does not depend on the machine running the hook: the win-64 target adds gxx_win-64.
EXPORT_COMMAND = [
    "pixi",
    "workspace",
    "export",
    "conda-environment",
    "-e",
    "default",
    "-n",
    "geconpy-dev",
    "-p",
    "linux-64",
]

# conda runs pip with the environment file's directory as its working directory (conda/env/installers/pip.py), so
# pixi's `-e .` would name conda_envs/ instead of the repository root.
PIXI_EDITABLE = "-e ."
REPO_EDITABLE = "-e .."

CONDA_ITEM = "- "
PIP_ITEM = "  - "

PACKAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def normalized_name(spec: str) -> str:
    """Return the PEP 503 normalized package name at the start of a conda or pip requirement."""
    match = PACKAGE_NAME.match(spec)
    if match is None:
        raise ValueError(f"No package name in requirement {spec!r}")
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def rewrite_pip_section(exported: str) -> str:
    """Point the editable install at the repository root and drop pip requirements the conda section supplies."""
    lines = exported.rstrip("\n").splitlines()

    # pixi repeats [project].dependencies in the pip section, and conda installs that section with `pip install -U`,
    # which could replace a conda-forge build with a PyPI wheel.
    conda_packages = {normalized_name(line.removeprefix(CONDA_ITEM)) for line in lines if line.startswith(CONDA_ITEM)}

    if f"{PIP_ITEM}{PIXI_EDITABLE}" not in lines:
        raise ValueError(f"pixi's export no longer contains the editable install line {PIXI_EDITABLE!r}")

    rewritten = []
    for line in lines:
        requirement = line.removeprefix(PIP_ITEM)
        if not line.startswith(PIP_ITEM):
            rewritten.append(line)
        elif requirement == PIXI_EDITABLE:
            rewritten.append(f"{PIP_ITEM}{REPO_EDITABLE}")
        elif normalized_name(requirement) not in conda_packages:
            rewritten.append(line)
    return "\n".join(rewritten) + "\n"


def main() -> None:
    exported = subprocess.run(EXPORT_COMMAND, cwd=REPO, check=True, stdout=subprocess.PIPE, text=True).stdout
    OUTPUT.write_text(rewrite_pip_section(exported), encoding="utf-8")


if __name__ == "__main__":
    main()
