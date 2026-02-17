import pytest

from gEconpy.parser.error_catalog import ErrorCode, format_error_help, get_error_info


class TestErrorCatalog:
    def test_catalog_entries_are_valid(self):
        for code in ErrorCode:
            assert code.title
            assert code.explanation
            assert code.info.common_causes
            assert code.info.fixes

    def test_get_error_info_returns_none_for_unknown(self):
        assert get_error_info("INVALID") is None

    def test_get_error_info_works_with_enum(self):
        info = get_error_info(ErrorCode.E001)
        assert info is not None
        assert info.title == "Missing semicolon"

    def test_get_error_info_works_with_string(self):
        info = get_error_info("E001")
        assert info is not None
        assert info.title == "Missing semicolon"

    def test_format_error_help(self):
        help_text = format_error_help("E001")
        assert "E001" in help_text
        assert "Common causes:" in help_text
        assert "How to fix:" in help_text

    def test_format_error_help_with_enum(self):
        help_text = format_error_help(ErrorCode.E001)
        assert "E001" in help_text
        assert "Missing semicolon" in help_text

    def test_format_error_help_unknown_returns_empty(self):
        assert format_error_help("INVALID") == ""
