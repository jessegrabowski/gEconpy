import pytest
import sympy as sp

from gEconpy.model.block import Block, dispatch_block
from gEconpy.model.block import registry as registry_mod
from gEconpy.model.block.registry import register_block


def _stub_block(detect_result):
    class _Stub(Block):
        @classmethod
        def detect(cls, *args, **kwargs):
            return detect_result

    return _Stub


def test_dispatch_falls_back_for_empty_block():
    assert type(dispatch_block(name="EMPTY")) is Block


class TestRegistryMechanism:
    def test_register_block_appends_subclass(self, monkeypatch):
        monkeypatch.setattr(registry_mod, "_REGISTRY", [])
        stub = _stub_block(detect_result=False)
        register_block(stub)
        assert stub in registry_mod._REGISTRY

    def test_register_block_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(registry_mod, "_REGISTRY", [])
        stub = _stub_block(detect_result=False)
        register_block(stub)
        register_block(stub)
        assert registry_mod._REGISTRY.count(stub) == 1

    def test_register_block_rejects_non_block(self):
        class _NotABlock:
            pass

        with pytest.raises(TypeError, match="Block subclass"):
            register_block(_NotABlock)

    def test_dispatch_returns_first_matching_subclass(self, monkeypatch):
        always, never = _stub_block(detect_result=True), _stub_block(detect_result=False)
        monkeypatch.setattr(registry_mod, "_REGISTRY", [always, never])
        assert isinstance(dispatch_block(name="X"), always)

    def test_dispatch_order_matters(self, monkeypatch):
        """Registration order is precedence: when two subclasses match, the earlier one wins."""
        first, second = _stub_block(detect_result=True), _stub_block(detect_result=True)

        monkeypatch.setattr(registry_mod, "_REGISTRY", [first, second])
        assert isinstance(dispatch_block(name="X"), first)

        monkeypatch.setattr(registry_mod, "_REGISTRY", [second, first])
        assert isinstance(dispatch_block(name="X"), second)


def test_detect_receives_the_dicts_given_to_dispatch(monkeypatch):
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
    equation_flags = {0: {"is_calibrating": False}, 2: {"is_calibrating": False}}
    dispatch_block(name="X", constraints=constraints, identities=identities, equation_flags=equation_flags)

    assert seen["constraints"] is constraints
    assert seen["objective"] is None
    assert seen["identities"] is identities
