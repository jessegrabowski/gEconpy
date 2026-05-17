import pytest
import sympy as sp

from gEconpy.model.block import Block, dispatch_block
from gEconpy.model.block import registry as registry_mod
from gEconpy.model.block.registry import register_block


def _stub_block(detect_result):
    """Build a fresh, unregistered Block subclass whose ``detect`` returns a constant."""

    class _Stub(Block):
        @classmethod
        def detect(cls, *args, **kwargs):
            return detect_result

    return _Stub


class TestDispatchAPI:
    """Drop-in replacement contract for ``Block(...)``."""

    def test_dispatch_falls_back_for_empty_block(self):
        """A block with no matchable structure must fall back to the base :class:`Block`."""
        b = dispatch_block(name="EMPTY")
        assert type(b) is Block


class TestRegistryMechanism:
    """Subclass registration via :func:`register_block` and dispatch ordering."""

    def test_register_block_appends_subclass(self, monkeypatch):
        """The decorator appends the class to the registry."""
        monkeypatch.setattr(registry_mod, "_REGISTRY", [])
        stub = _stub_block(detect_result=False)
        register_block(stub)
        assert stub in registry_mod._REGISTRY

    def test_register_block_is_idempotent(self, monkeypatch):
        """Re-registering the same class is a no-op (no duplicate entries)."""
        monkeypatch.setattr(registry_mod, "_REGISTRY", [])
        stub = _stub_block(detect_result=False)
        register_block(stub)
        register_block(stub)
        assert registry_mod._REGISTRY.count(stub) == 1

    def test_register_block_rejects_non_block(self):
        """register_block must reject anything that is not a Block subclass."""

        class _NotABlock:
            pass

        with pytest.raises(TypeError, match="Block subclass"):
            register_block(_NotABlock)

    def test_dispatch_returns_first_matching_subclass(self, monkeypatch):
        """The dispatcher walks the registry in order and constructs the first match."""
        always, never = _stub_block(detect_result=True), _stub_block(detect_result=False)
        monkeypatch.setattr(registry_mod, "_REGISTRY", [always, never])
        assert isinstance(dispatch_block(name="X"), always)

    def test_dispatch_order_matters(self, monkeypatch):
        """If two subclasses both match, the earlier one wins (registration order is precedence)."""
        first, second = _stub_block(detect_result=True), _stub_block(detect_result=True)

        monkeypatch.setattr(registry_mod, "_REGISTRY", [first, second])
        assert isinstance(dispatch_block(name="X"), first)

        monkeypatch.setattr(registry_mod, "_REGISTRY", [second, first])
        assert isinstance(dispatch_block(name="X"), second)


class TestDispatchDetectArgs:
    """The dispatcher passes the constraint/objective/identity dicts through to each subclass's ``detect``."""

    def test_detect_receives_kwargs(self, monkeypatch):
        """detect() sees the exact dicts dispatch_block was given."""
        seen = {}

        class _Recorder(Block):
            @classmethod
            def detect(cls, constraints, objective, identities):
                seen.update(constraints=constraints, objective=objective, identities=identities)
                return False

        monkeypatch.setattr(registry_mod, "_REGISTRY", [_Recorder])

        Y, x = sp.symbols("Y x")
        constraints = {0: sp.Eq(Y, x)}
        identities = {2: sp.Eq(x, 1)}
        # equation_flags must have entries for every keyed equation; objective + controls are paired or both absent.
        equation_flags = {0: {"is_calibrating": False}, 2: {"is_calibrating": False}}
        dispatch_block(name="X", constraints=constraints, identities=identities, equation_flags=equation_flags)
        assert seen["constraints"] is constraints
        assert seen["objective"] is None
        assert seen["identities"] is identities
