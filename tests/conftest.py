from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
TESTS_ROOT = PROJECT_ROOT / "tests"
RESOURCES = TESTS_ROOT / "_resources"
TEST_GCNS = RESOURCES / "test_gcns"
ERROR_GCNS = RESOURCES / "error_gcns"
