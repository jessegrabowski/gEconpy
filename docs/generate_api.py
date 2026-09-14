import importlib
import inspect
import pkgutil
import sys

from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
API_DIR = DOCS_DIR / "source" / "api"

PACKAGES = ["gEconpy"]
SKIP_SUFFIXES = ("._version",)


def iter_modules(package_name: str, on_error) -> list[str]:
    package = importlib.import_module(package_name)
    names = [package_name]
    for info in pkgutil.walk_packages(package.__path__, f"{package_name}.", onerror=on_error):
        if info.name.endswith(SKIP_SUFFIXES):
            continue
        names.append(info.name)
    return names


def member_kind(obj) -> str | None:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(inspect.unwrap(obj)):
        return "function"
    return None


def public_members(module) -> tuple[list[str], list[str], list[str]]:
    """Split a module's documented names into own classes, own functions and re-exports.

    A module that declares ``__all__`` documents exactly that list. One that does not documents every class and
    function it defines whose name has no leading underscore.
    """
    if hasattr(module, "__all__"):
        candidates = [(name, getattr(module, name)) for name in module.__all__]
    else:
        candidates = [
            (name, obj)
            for name, obj in vars(module).items()
            if not name.startswith("_") and getattr(obj, "__module__", None) == module.__name__
        ]

    classes, functions, reexported = [], [], []
    for name, obj in candidates:
        kind = member_kind(obj)
        if kind is None:
            continue
        if getattr(obj, "__module__", None) == module.__name__:
            (classes if kind == "class" else functions).append(name)
        else:
            # A re-export links to its canonical page instead of getting a second description. The
            # path is relative to this module, which is the page's currentmodule.
            canonical = f"{obj.__module__}.{name}"
            reexported.append(canonical.removeprefix(f"{module.__name__}."))
    return sorted(classes), sorted(functions), sorted(reexported)


def rst_for_module(module_name: str, classes, functions, reexported=()) -> str:
    lines = [module_name, "=" * len(module_name), "", f".. currentmodule:: {module_name}", ""]
    for rubric, names in (("Classes", classes), ("Functions", functions), ("Re-exported", reexported)):
        if not names:
            continue
        lines += [f".. rubric:: {rubric}", "", ".. autosummary::", ""]
        lines += [f"    {name}" for name in names]
        lines.append("")
    # One page per module builds far faster than a page per object and still anchors every class, method
    # and function for deep linking. Naming the members keeps a package page from describing its re-exports
    # a second time, and undoc-members keeps a member without a docstring visible.
    own = sorted(classes) + sorted(functions)
    if own:
        lines += [f".. automodule:: {module_name}", f"    :members: {', '.join(own)}", "    :undoc-members:", ""]
    return "\n".join(lines)


def write_if_changed(path: Path, text: str) -> None:
    """Skip rewriting an identical page, since a bumped mtime makes Sphinx rebuild everything that links it."""
    if path.exists() and path.read_text() == text:
        return
    path.write_text(text)


def write_module_pages(package_name: str, failures: list[tuple[str, str]], stale: set[Path]) -> list[str]:
    """Write one page per importable module below ``package_name`` and return the module names that have a page."""

    def on_walk_error(name: str) -> None:
        exc = sys.exc_info()[1]
        failures.append((name, f"{type(exc).__name__}: {exc}"))

    documented = []
    for module_name in iter_modules(package_name, on_walk_error):
        if module_name == package_name:
            continue
        page = API_DIR / f"{module_name}.rst"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            failures.append((module_name, f"{type(exc).__name__}: {exc}"))
            # A module that fails to import keeps its existing page rather than being pruned.
            if page.exists():
                stale.discard(page)
                documented.append(module_name)
            continue
        members = public_members(module)
        if not any(members):
            continue
        write_if_changed(page, rst_for_module(module_name, *members))
        stale.discard(page)
        documented.append(module_name)
    return documented


def write_package_page(package_name: str, modules: list[str], stale: set[Path]) -> None:
    body = rst_for_module(package_name, *public_members(importlib.import_module(package_name)))
    toc = ["", ".. rubric:: Submodules", "", ".. toctree::", "    :maxdepth: 1", ""]
    toc += [f"    {name} <{name}>" for name in sorted(modules)]
    page = API_DIR / f"{package_name}.rst"
    write_if_changed(page, body + "\n".join(toc) + "\n")
    stale.discard(page)


def write_index(stale: set[Path]) -> None:
    index = [".. _api:", "", "API Reference", "=============", "", ".. toctree::", "    :maxdepth: 2", ""]
    index += [f"    {pkg} <{pkg}>" for pkg in PACKAGES]
    page = API_DIR / "index.rst"
    write_if_changed(page, "\n".join(index) + "\n")
    stale.discard(page)


def main() -> int:
    API_DIR.mkdir(parents=True, exist_ok=True)
    stale = set(API_DIR.glob("*.rst"))
    failures: list[tuple[str, str]] = []

    written = 0
    for package_name in PACKAGES:
        modules = write_module_pages(package_name, failures, stale)
        write_package_page(package_name, modules, stale)
        written += len(modules) + 1
    write_index(stale)

    for page in stale:
        page.unlink()

    print(f"Wrote {written} module pages under {API_DIR}")
    if failures:
        print(f"\n{len(failures)} module(s) failed to import:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        print("\nTheir pages are unchanged. Run in the docs environment: pixi run -e docs docs-api")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
