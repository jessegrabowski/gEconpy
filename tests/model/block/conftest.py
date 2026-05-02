import pytest

from gEconpy.model.block import registry as registry_mod


@pytest.fixture
def disable_dispatch(monkeypatch):
    """Force the registry to dispatch nothing — every block becomes the base :class:`Block`."""
    monkeypatch.setattr(registry_mod, "_REGISTRY", [])
