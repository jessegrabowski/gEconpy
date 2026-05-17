import sympy as sp

from gEconpy.model.block import Block, dispatch_block
from gEconpy.model.block import registry as registry_mod
from gEconpy.model.block.registry import register_block


class TestDispatchAPI:
    """Drop-in replacement contract for ``Block(...)``."""

    def test_dispatch_falls_back_for_empty_block(self):
        """A block with no matchable structure must fall back to the base :class:`Block`."""
        b = dispatch_block(name="EMPTY")
        assert type(b) is Block

    def test_block_import_still_works(self):
        """Existing ``from gEconpy.model.block import Block`` must continue to resolve."""
        from gEconpy.model.block import Block as B  # noqa: PLC0415

        assert B is Block


class TestRegistryMechanism:
    """Subclass registration via :func:`register_block` and dispatch ordering."""

    def test_register_block_appends_subclass(self, monkeypatch):
        """The decorator appends the class to the registry in declaration order."""
        monkeypatch.setattr(registry_mod, "_REGISTRY", [])

        @register_block
        class _FakeBlock(Block):
            @classmethod
            def detect(cls, constraints, objective, identities):
                return False

        assert _FakeBlock in registry_mod._REGISTRY

    def test_register_block_is_idempotent(self, monkeypatch):
        """Re-registering the same class is a no-op (no duplicate entries)."""
        monkeypatch.setattr(registry_mod, "_REGISTRY", [])

        @register_block
        class _FakeBlock(Block):
            @classmethod
            def detect(cls, constraints, objective, identities):
                return False

        register_block(_FakeBlock)
        assert registry_mod._REGISTRY.count(_FakeBlock) == 1

    def test_dispatch_returns_first_matching_subclass(self, monkeypatch):
        """The dispatcher walks the registry in order and constructs the first match."""

        class _AlwaysMatch(Block):
            @classmethod
            def detect(cls, constraints, objective, identities):
                return True

        class _NeverMatch(Block):
            @classmethod
            def detect(cls, constraints, objective, identities):
                return False

        monkeypatch.setattr(registry_mod, "_REGISTRY", [_AlwaysMatch, _NeverMatch])
        b = dispatch_block(name="X")
        assert isinstance(b, _AlwaysMatch)

    def test_dispatch_order_matters(self, monkeypatch):
        """If two subclasses both match, the earlier one wins (registration order is precedence)."""

        class _FirstMatch(Block):
            @classmethod
            def detect(cls, constraints, objective, identities):
                return True

        class _SecondMatch(Block):
            @classmethod
            def detect(cls, constraints, objective, identities):
                return True

        monkeypatch.setattr(registry_mod, "_REGISTRY", [_FirstMatch, _SecondMatch])
        assert isinstance(dispatch_block(name="X"), _FirstMatch)

        monkeypatch.setattr(registry_mod, "_REGISTRY", [_SecondMatch, _FirstMatch])
        assert isinstance(dispatch_block(name="X"), _SecondMatch)


class TestDispatchDetectArgs:
    """The dispatcher passes the same constraint/objective/identity dicts to every subclass's ``detect``."""

    def test_detect_receives_kwargs(self, monkeypatch):
        seen = {}

        class _Recorder(Block):
            @classmethod
            def detect(cls, constraints, objective, identities):
                seen["constraints"] = constraints
                seen["objective"] = objective
                seen["identities"] = identities
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
