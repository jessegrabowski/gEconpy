from collections.abc import Callable
from difflib import unified_diff
from pathlib import Path

import pytest

from gEconpy.exceptions import GCNValidationError
from gEconpy.parser.errors import GCNParseError
from gEconpy.parser.formatting import ErrorFormatter
from gEconpy.parser.loader import load_gcn_file
from gEconpy.parser.preprocessor import preprocess
from tests.conftest import ERROR_GCNS as ERROR_GCNS_DIR


def golden_cases(prefix_filter: Callable[[str], bool]) -> list[Path]:
    """Collect the error ``.gcn`` fixtures that have a matching ``.expected`` file and pass the name filter."""
    return [
        gcn_file
        for gcn_file in sorted(ERROR_GCNS_DIR.glob("*.gcn"))
        if gcn_file.with_suffix(".expected").exists() and prefix_filter(gcn_file.name)
    ]


PARSE_ERROR_CASES = golden_cases(lambda name: not name.startswith("V"))
VALIDATION_ERROR_CASES = golden_cases(lambda name: name.startswith("V"))


@pytest.mark.parametrize("gcn_file", PARSE_ERROR_CASES, ids=lambda path: path.name)
def test_parse_error_output_matches_golden_file(gcn_file):
    content = gcn_file.read_text(encoding="utf-8")
    formatter = ErrorFormatter(use_color=False)

    with pytest.raises(GCNParseError) as exc_info:
        preprocess(content, validate=True, filename=gcn_file.name)

    assert_matches_golden(formatter.format_error(exc_info.value, content), gcn_file)


@pytest.mark.parametrize("gcn_file", VALIDATION_ERROR_CASES, ids=lambda path: path.name)
def test_validation_error_output_matches_golden_file(gcn_file):
    with pytest.raises(GCNValidationError) as exc_info:
        load_gcn_file(str(gcn_file))

    assert_matches_golden(str(exc_info.value), gcn_file)


def assert_matches_golden(actual_output: str, gcn_file: Path) -> None:
    expected_file = gcn_file.with_suffix(".expected")
    expected_output = expected_file.read_text(encoding="utf-8").rstrip("\n")
    if actual_output == expected_output:
        return

    diff = "\n".join(
        unified_diff(
            expected_output.splitlines(), actual_output.splitlines(), expected_file.name, "actual", lineterm=""
        )
    )
    raise AssertionError(
        f"Error output for {gcn_file.name} does not match {expected_file.name}. "
        f"To regenerate the expected files, run: python scripts/regenerate_expected_gcn_errors.py\n{diff}"
    )
