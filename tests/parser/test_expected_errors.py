import sys

from difflib import unified_diff
from pathlib import Path

import pytest

from gEconpy.parser.errors import GCNParseError
from gEconpy.parser.formatting import ErrorFormatter
from gEconpy.parser.preprocessor import preprocess

ERROR_GCNS_DIR = Path(__file__).parent.parent / "_resources" / "error_gcns"


def get_error_test_cases():
    """Generate test cases from .gcn files with matching .expected files."""
    cases = []
    for gcn_file in sorted(ERROR_GCNS_DIR.glob("*.gcn")):
        expected_file = gcn_file.with_suffix(".expected")
        if expected_file.exists():
            cases.append((gcn_file.name, gcn_file, expected_file))
    return cases


@pytest.mark.parametrize(
    "name,gcn_file,expected_file",
    get_error_test_cases(),
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_error_output_matches_golden_file(name, gcn_file, expected_file):
    content = gcn_file.read_text()
    formatter = ErrorFormatter(use_color=False)

    with pytest.raises(GCNParseError) as exc_info:
        preprocess(content, validate=True, filename=gcn_file.name)

    actual_output = formatter.format_error(exc_info.value, content)
    expected_output = expected_file.read_text().rstrip("\n")

    if actual_output != expected_output:
        actual_lines = actual_output.splitlines()
        expected_lines = expected_output.splitlines()

        diff = unified_diff(actual_lines, expected_lines)
        for line in diff:
            if line.startswith("-"):
                print(f"\033[31m{line}\033[0m")
            elif line.startswith("+"):
                print(f"\033[32m{line}\033[0m")
            else:
                print(line)
        raise AssertionError(
            f"Error output for {name} doesn't match expected.\n"
            f"To update expected file, run the regeneration script in the module docstring."
        )
