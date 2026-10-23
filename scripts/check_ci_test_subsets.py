import ast
import shlex
import sys

from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "run_tests.yml"

# Adapted from PyMC's scripts/check_all_tests_are_covered.py. Tests are counted one by one rather than per file,
# because the matrix splits tests/model/test_model.py across jobs by node id with ``file::test`` and ``--deselect``.

# The Ubuntu jobs are the correctness sweep and must run every test once. The Windows jobs are a smoke subset, see
# the comment above them in the workflow, so they are only checked for names that do not exist and double runs.
FULL_COVERAGE_OS = "ubuntu-latest"


def test_nodes(test_file: Path) -> set[str]:
    """Return the node ids of the tests in ``test_file``, as ``file::test`` or ``file::Class::test``."""
    relative = test_file.relative_to(REPO).as_posix()
    tree = ast.parse(test_file.read_text(encoding="utf-8"))

    nodes = set()
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef) and statement.name.startswith("test"):
            nodes.add(f"{relative}::{statement.name}")
        elif isinstance(statement, ast.ClassDef) and statement.name.startswith("Test"):
            nodes.update(
                f"{relative}::{statement.name}::{method.name}"
                for method in statement.body
                if isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef) and method.name.startswith("test")
            )
    return nodes


def all_test_nodes() -> dict[str, set[str]]:
    """Return every test node in the repository, keyed by its file's path relative to the repository root."""
    return {
        test_file.relative_to(REPO).as_posix(): test_nodes(test_file)
        for test_file in sorted((REPO / "tests").glob("**/test_*.py"))
    }


def parse_subset(subset: str) -> tuple[list[str], list[str], list[str]]:
    """Split a pytest argument string into selected paths, ignored paths, and deselected node ids."""
    selected, ignored, deselected = [], [], []
    tokens = iter(shlex.split(subset))
    for token in tokens:
        if token.startswith("--ignore"):
            ignored.append(token.partition("=")[2] or next(tokens))
        elif token.startswith("--deselect"):
            deselected.append(token.partition("=")[2] or next(tokens))
        elif not token.startswith("-"):
            selected.append(token)
    return selected, ignored, deselected


def under(path: str, prefix: str) -> bool:
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(f"{prefix}/")


def missing_names(names: list[str], nodes_by_file: dict[str, set[str]]) -> list[str]:
    """Return the paths and node ids in ``names`` that match no test file, directory, or test."""
    missing = []
    for name in names:
        path, _, node = name.partition("::")
        if node:
            if f"{path}::{node.split('[')[0]}" not in nodes_by_file.get(path, set()):
                missing.append(name)
        elif not any(under(test_file, path) for test_file in nodes_by_file):
            missing.append(name)
    return missing


def nodes_run_by(subset: str, nodes_by_file: dict[str, set[str]]) -> set[str]:
    """Return the test nodes one job runs."""
    selected, ignored, deselected = parse_subset(subset)
    run = set()
    for name in selected:
        path, _, node = name.partition("::")
        if node:
            run.add(name)
            continue
        for test_file, nodes in nodes_by_file.items():
            if under(test_file, path) and not any(under(test_file, ignore) for ignore in ignored):
                run.update(nodes)
    return run - set(deselected)


def main() -> int:
    """Print every problem with the CI test subsets and return 1 when there is one."""
    nodes_by_file = all_test_nodes()
    every_node = set().union(*nodes_by_file.values())
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["test"]["strategy"]["matrix"]["include"]

    problems = []
    runs_by_os: dict[str, Counter] = {}
    for job in jobs:
        selected, ignored, deselected = parse_subset(job["test-subset"])
        problems.extend(
            f"{job['subset-name']}: '{name}' matches no test file, directory, or test"
            for name in missing_names(selected + ignored + deselected, nodes_by_file)
        )
        runs_by_os.setdefault(job["os"], Counter()).update(nodes_run_by(job["test-subset"], nodes_by_file))

    for os_name, runs in sorted(runs_by_os.items()):
        problems.extend(f"{os_name}: {node} runs in {count} jobs" for node, count in sorted(runs.items()) if count > 1)
        if os_name == FULL_COVERAGE_OS:
            problems.extend(f"{os_name}: {node} runs in no job" for node in sorted(every_node - set(runs)))

    for _problem in problems:
        pass
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
