import pytest

from gEconpy.parser.error_catalog import ErrorCode, format_error_help, get_error_info


def test_every_catalog_entry_is_complete():
    for code in ErrorCode:
        assert code.title
        assert code.explanation
        assert code.info.common_causes
        assert code.info.fixes


@pytest.mark.parametrize("code", ["E001", ErrorCode.E001], ids=["string", "enum"])
def test_get_error_info(code):
    info = get_error_info(code)
    assert info is not None
    assert info.title == "Missing semicolon"


def test_get_error_info_returns_none_for_unknown():
    assert get_error_info("INVALID") is None


@pytest.mark.parametrize("code", ["E001", ErrorCode.E001], ids=["string", "enum"])
def test_format_error_help(code):
    help_text = format_error_help(code)
    assert help_text.startswith("E001: Missing semicolon")
    assert "Common causes:" in help_text
    assert "How to fix:" in help_text


def test_format_error_help_unknown_returns_empty():
    assert format_error_help("INVALID") == ""
