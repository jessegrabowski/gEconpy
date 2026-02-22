from pathlib import Path

from gEconpy.exceptions import GCNValidationError
from gEconpy.parser.errors import GCNParseError
from gEconpy.parser.formatting import ErrorFormatter
from gEconpy.parser.loader import load_gcn_file
from gEconpy.parser.preprocessor import preprocess


def regenerate_expected_files() -> None:
    """Regenerate all expected error output files."""
    # Find the error_gcns directory relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    error_dir = project_root / "tests" / "_resources" / "error_gcns"

    if not error_dir.exists():
        msg = f"Error directory not found: {error_dir}"
        raise FileNotFoundError(msg)

    formatter = ErrorFormatter(use_color=False)
    updated = 0
    warnings = []

    for gcn_file in sorted(error_dir.glob("*.gcn")):
        expected_file = gcn_file.with_suffix(".expected")
        content = gcn_file.read_text()
        filename = gcn_file.name

        # Determine if this is a validation error (Vxxx) or parse error (Exxx)
        is_validation_error = filename.startswith("V")

        try:
            if is_validation_error:
                # Validation errors occur during model building, not parsing
                load_gcn_file(str(gcn_file))
                warnings.append(f"WARNING: {filename} did not raise an error!")
            else:
                # Parse errors occur during preprocessing
                preprocess(content, validate=True, filename=filename)
                warnings.append(f"WARNING: {filename} did not raise an error!")

        except GCNValidationError as e:
            # Validation errors have their own formatting
            output = str(e)
            expected_file.write_text(output + "\n")
            line = e.location.line if e.location else "?"
            col = e.location.column if e.location else "?"
            code = e.error_code or "(none)"
            updated += 1
            print(f"Updated {expected_file.name} - line {line}, col {col}, code: {code}")

        except GCNParseError as e:
            output = formatter.format_error(e, content)
            expected_file.write_text(output + "\n")
            line = e.location.line if e.location else "?"
            col = e.location.column if e.location else "?"
            code = e.code or "(none)"
            updated += 1
            print(f"Updated {expected_file.name} - line {line}, col {col}, code: {code}")

        except Exception as ex:
            warnings.append(f"ERROR: {filename}: {type(ex).__name__}: {ex}")

    print(f"\nRegenerated {updated} expected files.")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  {warning}")


if __name__ == "__main__":
    regenerate_expected_files()
