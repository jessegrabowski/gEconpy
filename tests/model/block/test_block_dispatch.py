"""Tests for the block dispatch system and CobbDouglasBlock.

Phase 1 deliverable for the special blocks roadmap (audits/special_blocks/roadmap.md).

Validates that:
  * The parser dispatches to CobbDouglasBlock for canonical Cobb-Douglas
    production blocks, and falls back to Block for everything else.
  * Detection is conservative (no false positives on near-CD forms).
  * The closed-form FOCs produce a policy function (T, R) identical to the
    general Block path at machine epsilon.
"""

import gc

import numpy as np
import pytest
import sympy as sp

from gEconpy import model_from_gcn
from gEconpy.model.block import Block, dispatch_block
from gEconpy.model.block import registry as registry_mod
from gEconpy.model.block.production import CobbDouglasBlock, _match_cobb_douglas_constraint
from gEconpy.parser.loader import load_gcn_file
from gEconpy.solvers.cycle_reduction import solve_policy_function_with_cycle_reduction

RBC_PATH = "gEconpy/data/GCN Files/RBC.gcn"


@pytest.fixture
def disable_dispatch(monkeypatch):
    """Force the registry to dispatch nothing — all blocks become base Block."""
    monkeypatch.setattr(registry_mod, "_REGISTRY", [])


class TestDispatchOnRBC:
    def test_firm_is_dispatched(self):
        """RBC's FIRM block has a canonical CD constraint — must dispatch."""
        prims = load_gcn_file(RBC_PATH, simplify_blocks=True)
        firm = prims.block_dict["FIRM"]
        assert isinstance(firm, CobbDouglasBlock)

    def test_non_firm_blocks_fall_back(self):
        """HOUSEHOLD and TECHNOLOGY_SHOCKS must NOT dispatch — they're not CD."""
        prims = load_gcn_file(RBC_PATH, simplify_blocks=True)
        for name in ("HOUSEHOLD", "TECHNOLOGY_SHOCKS"):
            blk = prims.block_dict[name]
            assert type(blk) is Block, f"{name} should be base Block, got {type(blk).__name__}"

    def test_closed_form_focs_in_equations(self):
        """The FIRM block's FOCs must be in closed-form Y/K and Y/L form, not the chain-rule expansion."""
        m = model_from_gcn(RBC_PATH, verbose=False)
        eq_strings = [str(eq) for eq in m.equations]
        # eq6 must contain Y_t/K_t-1, NOT K_t-1**(alpha - 1).
        assert any("Y_t/K_t-1" in s for s in eq_strings), f"Expected closed-form Y/K FOC, got equations: {eq_strings}"
        # No chain-rule K**(alpha - 1) form anywhere.
        assert not any("K_t-1**(alpha - 1)" in s for s in eq_strings), (
            f"Chain-rule expansion leaked through: {eq_strings}"
        )


class TestPolicyEquivalence:
    """The dispatched path must produce the same policy function as the general Block path, to machine epsilon."""

    @pytest.mark.usefixtures("disable_dispatch")
    def test_rbc_policy_matches_baseline(self):
        """Confirm dispatch-disabled run solves cleanly (baseline for the equivalence check below)."""
        m_base = model_from_gcn(RBC_PATH, verbose=False)
        A, B, C, D = m_base.linearize_model(verbose=False)
        T_base, R_base, _, _ = solve_policy_function_with_cycle_reduction(A, B, C, D, 1000, 1e-12, False)
        assert T_base is not None
        assert R_base is not None

    def test_rbc_policy_matches_dispatched(self):
        """Confirm the dispatched path solves cleanly."""
        m = model_from_gcn(RBC_PATH, verbose=False)
        A, B, C, D = m.linearize_model(verbose=False)
        T, R, _, _ = solve_policy_function_with_cycle_reduction(A, B, C, D, 1000, 1e-12, False)
        assert T is not None
        assert R is not None
        assert np.all(np.isfinite(T))
        assert np.all(np.isfinite(R))

    def test_rbc_policy_equivalence(self, monkeypatch):
        """Compute T, R with and without dispatch and compare element-wise."""
        # Dispatched run
        m = model_from_gcn(RBC_PATH, verbose=False)
        A, B, C, D = m.linearize_model(verbose=False)
        T, R, _, _ = solve_policy_function_with_cycle_reduction(A, B, C, D, 1000, 1e-12, False)

        # Disable dispatch and re-run
        monkeypatch.setattr(registry_mod, "_REGISTRY", [])
        gc.collect()
        m_b = model_from_gcn(RBC_PATH, verbose=False)
        A_b, B_b, C_b, D_b = m_b.linearize_model(verbose=False)
        T_b, R_b, _, _ = solve_policy_function_with_cycle_reduction(A_b, B_b, C_b, D_b, 1000, 1e-12, False)

        assert np.max(np.abs(T - T_b)) < 1e-12, f"Transition matrix T differs: max |Δ| = {np.max(np.abs(T - T_b)):.3e}"
        assert np.max(np.abs(R - R_b)) < 1e-12, f"Impact matrix R differs: max |Δ| = {np.max(np.abs(R - R_b)):.3e}"


class TestDetectionConservatism:
    """Detection must be conservative: false positives are bugs."""

    def test_rejects_no_constraint(self):
        """No constraint → no match (CD requires a production constraint)."""
        assert _match_cobb_douglas_constraint(None) is None
        assert _match_cobb_douglas_constraint({}) is None

    def test_rejects_two_constraints(self):
        """Multiple constraints → no match (CD has exactly one)."""
        Y, A, x1, x2, a1 = sp.symbols("Y A x1 x2 a1")
        constraints = {
            0: sp.Eq(Y, A * x1**a1 * x2 ** (1 - a1)),
            1: sp.Eq(Y, A * x1**a1 * x2 ** (1 - a1)),
        }
        assert _match_cobb_douglas_constraint(constraints) is None

    def test_rejects_non_cd_constraint(self):
        """Constraint that doesn't structurally match CD → no match."""
        Y, A, x1, x2 = sp.symbols("Y A x1 x2")
        # Linear production — not CD
        constraints = {0: sp.Eq(Y, A * (x1 + x2))}
        assert _match_cobb_douglas_constraint(constraints) is None

    def test_rejects_exponent_zero(self):
        """Wild excludes a_i=0 (degenerate input collapses to constant)."""
        Y, A, x1, x2 = sp.symbols("Y A x1 x2")
        constraints = {0: sp.Eq(Y, A * x1**0 * x2**1)}  # = A * x2
        assert _match_cobb_douglas_constraint(constraints) is None

    def test_accepts_crs_form(self):
        """The constant-returns-to-scale Y = A*x1^a1*x2^(1-a1) form must match."""
        Y, A, x1, x2, a1 = sp.symbols("Y A x1 x2 a1")
        constraints = {3: sp.Eq(Y, A * x1**a1 * x2 ** (1 - a1))}
        m = _match_cobb_douglas_constraint(constraints)
        assert m is not None
        assert m["Y"] == Y
        assert m["A"] == A
        assert m["idx"] == 3
        # Inputs come back as a list of (symbol, exponent) pairs in some order;
        # check both inputs are present and the exponents are recovered.
        inputs = dict(m["inputs"])
        assert set(inputs.keys()) == {x1, x2}
        assert inputs[x1] + inputs[x2] == 1  # a1 + (1 - a1) = 1

    def test_accepts_non_crs_form(self):
        """Decreasing returns to scale (a1 + a2 != 1) must still match — the closed-form FOC holds for any monomial."""
        Y, A, x1, x2, a1, a2 = sp.symbols("Y A x1 x2 a1 a2")
        constraints = {0: sp.Eq(Y, A * x1**a1 * x2**a2)}
        m = _match_cobb_douglas_constraint(constraints)
        assert m is not None
        assert m["Y"] == Y
        assert m["A"] == A
        inputs = dict(m["inputs"])
        assert set(inputs.keys()) == {x1, x2}
        assert {inputs[x1], inputs[x2]} == {a1, a2}

    @pytest.mark.parametrize("k", [1, 3, 10])
    def test_accepts_arbitrary_arity(self, k):
        """Arity-agnostic: Y = A * prod(x_i^a_i for i in 1..k) must match for any k >= 1."""
        Y, A = sp.symbols("Y A")
        xs = sp.symbols(f"x1:{k + 1}")  # x1, x2, ..., xk
        as_ = sp.symbols(f"a1:{k + 1}")  # a1, a2, ..., ak
        prod = A
        for x, a in zip(xs, as_, strict=True):
            prod = prod * x**a
        constraints = {0: sp.Eq(Y, prod)}
        m = _match_cobb_douglas_constraint(constraints)
        assert m is not None, f"k={k}: matcher returned None"
        assert m["Y"] == Y
        assert m["A"] == A
        assert len(m["inputs"]) == k
        recovered = dict(m["inputs"])
        assert set(recovered.keys()) == set(xs)
        for x, a in zip(xs, as_, strict=True):
            assert recovered[x] == a, f"k={k}: input {x} got exponent {recovered[x]}, expected {a}"

    def test_rejects_bare_input_ambiguous_with_A(self):
        """Y = A * x is ambiguous: both A and x are bare Symbols. Reject conservatively."""
        Y, A, x = sp.symbols("Y A x")
        constraints = {0: sp.Eq(Y, A * x)}
        assert _match_cobb_douglas_constraint(constraints) is None

    def test_rejects_extra_constant_factor(self):
        """Y = 2 * A * x1^a1 * x2^a2 has an extra numerical coefficient; not canonical CD."""
        Y, A, x1, x2, a1, a2 = sp.symbols("Y A x1 x2 a1 a2")
        constraints = {0: sp.Eq(Y, 2 * A * x1**a1 * x2**a2)}
        assert _match_cobb_douglas_constraint(constraints) is None

    def test_rejects_duplicate_input(self):
        """Y = A * x^a * x^b has duplicate input bases — reject rather than silently merge into x^(a+b)."""
        Y, A, x, a, b = sp.symbols("Y A x a b")
        constraints = {0: sp.Eq(Y, A * x**a * x**b)}
        assert _match_cobb_douglas_constraint(constraints) is None

    def test_rejects_extra_additive_term(self):
        """Y = A*x1^a1*x2^a2 + adjustment_cost(x1) is NOT a pure monomial."""
        Y, A, x1, x2, a1, a2, phi = sp.symbols("Y A x1 x2 a1 a2 phi")
        constraints = {0: sp.Eq(Y, A * x1**a1 * x2**a2 + phi * x1**2)}
        assert _match_cobb_douglas_constraint(constraints) is None

    def test_detect_requires_objective(self):
        """An identity-only block must not dispatch even when the constraint matches CD shape."""
        Y, A, x1, x2, a1 = sp.symbols("Y A x1 x2 a1")
        constraints = {0: sp.Eq(Y, A * x1**a1 * x2 ** (1 - a1))}
        assert CobbDouglasBlock.detect(constraints, objective=None, identities=None) is False
