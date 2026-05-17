import gc

import numpy as np
import pytest
import sympy as sp

from gEconpy import model_from_gcn
from gEconpy.model.block import Block
from gEconpy.model.block import registry as registry_mod
from gEconpy.model.block.ces import CESBlock, _match_ces_constraint
from gEconpy.parser.loader import load_gcn_file
from gEconpy.solvers.cycle_reduction import solve_policy_function_with_cycle_reduction

RBC_CES_PATH = "gEconpy/data/GCN Files/RBC_with_CES.gcn"


class TestDispatchOnRBCWithCES:
    def test_firm_is_dispatched(self):
        """RBC_with_CES's FIRM block has a canonical CES constraint — must dispatch."""
        prims = load_gcn_file(RBC_CES_PATH, simplify_blocks=True)
        firm = prims.block_dict["FIRM"]
        assert isinstance(firm, CESBlock)

    def test_non_firm_blocks_fall_back(self):
        """HOUSEHOLD and TECHNOLOGY_SHOCKS must NOT dispatch — they're not CES."""
        prims = load_gcn_file(RBC_CES_PATH, simplify_blocks=True)
        for name in ("HOUSEHOLD", "TECHNOLOGY_SHOCKS"):
            blk = prims.block_dict[name]
            assert type(blk) is Block, f"{name} should be base Block, got {type(blk).__name__}"

    def test_closed_form_focs_in_equations(self):
        """The FIRM block's FOCs must be in closed-form share*A^s*(Y/x)^(1-s) form, not the chain-rule expansion."""
        m = model_from_gcn(RBC_CES_PATH, verbose=False)
        eq_strings = [str(eq) for eq in m.equations]
        # Closed-form FOC for K should reference Y/K (and likewise Y/L for L).
        assert any("Y_t/K_t-1" in s for s in eq_strings), (
            f"Expected closed-form (Y/K)^(1-s) FOC, got equations: {eq_strings}"
        )
        assert any("Y_t/L_t" in s for s in eq_strings), (
            f"Expected closed-form (Y/L)^(1-s) FOC, got equations: {eq_strings}"
        )
        # Chain-rule signatures: the inner-sum Pow with the differentiated exponent (psi/(psi - 1) - 1), and the
        # decremented input exponent (-1 + (psi - 1)/psi). These appear ONLY when sympy.diff has been called on the
        # outer Pow over the inner sum — i.e., the dispatch path failed and the base Block's Lagrangian-diff fired.
        # The production constraint (eq5) legitimately contains the inner sum at outer_exp = psi/(psi-1), so that
        # exponent alone is not a leak signal; we look for the *decremented* form specifically.
        leak_markers = ("psi/(psi - 1) - 1", "-1 + (psi - 1)/psi")
        for marker in leak_markers:
            assert not any(marker in s for s in eq_strings), (
                f"Chain-rule expansion signature {marker!r} leaked through: {eq_strings}"
            )


class TestPolicyEquivalence:
    """The dispatched path must produce the same policy function as the general Block path, to machine epsilon."""

    @pytest.mark.usefixtures("disable_dispatch")
    def test_policy_matches_baseline(self):
        """Confirm dispatch-disabled run solves cleanly (baseline for the equivalence check below)."""
        m_base = model_from_gcn(RBC_CES_PATH, verbose=False)
        A, B, C, D = m_base.linearize_model(verbose=False)
        T_base, R_base, _, _ = solve_policy_function_with_cycle_reduction(A, B, C, D, 1000, 1e-12, False)
        assert T_base is not None
        assert R_base is not None

    def test_policy_matches_dispatched(self):
        """Confirm the dispatched path solves cleanly."""
        m = model_from_gcn(RBC_CES_PATH, verbose=False)
        A, B, C, D = m.linearize_model(verbose=False)
        T, R, _, _ = solve_policy_function_with_cycle_reduction(A, B, C, D, 1000, 1e-12, False)
        assert T is not None
        assert R is not None
        assert np.all(np.isfinite(T))
        assert np.all(np.isfinite(R))

    def test_policy_equivalence(self, monkeypatch):
        """Compute T, R with and without dispatch and compare element-wise."""
        m = model_from_gcn(RBC_CES_PATH, verbose=False)
        A, B, C, D = m.linearize_model(verbose=False)
        T, R, _, _ = solve_policy_function_with_cycle_reduction(A, B, C, D, 1000, 1e-12, False)

        monkeypatch.setattr(registry_mod, "_REGISTRY", [])
        gc.collect()
        m_b = model_from_gcn(RBC_CES_PATH, verbose=False)
        A_b, B_b, C_b, D_b = m_b.linearize_model(verbose=False)
        T_b, R_b, _, _ = solve_policy_function_with_cycle_reduction(A_b, B_b, C_b, D_b, 1000, 1e-12, False)

        assert np.max(np.abs(T - T_b)) < 1e-12, f"Transition matrix T differs: max |Δ| = {np.max(np.abs(T - T_b)):.3e}"
        assert np.max(np.abs(R - R_b)) < 1e-12, f"Impact matrix R differs: max |Δ| = {np.max(np.abs(R - R_b)):.3e}"


class TestDetectionConservatism:
    """Detection must be conservative: false positives are bugs."""

    def test_rejects_no_constraint(self):
        """No constraint → no match (CES requires a production constraint)."""
        assert _match_ces_constraint(None) is None
        assert _match_ces_constraint({}) is None

    def test_rejects_two_constraints(self):
        """Multiple constraints → no match (CES has exactly one)."""
        Y, A, x1, x2, alpha, psi = sp.symbols("Y A x1 x2 alpha psi")
        s = (psi - 1) / psi
        inner = alpha ** (1 / psi) * x1**s + (1 - alpha) ** (1 / psi) * x2**s
        constraint = sp.Eq(Y, A * inner ** (1 / s))
        assert _match_ces_constraint({0: constraint, 1: constraint}) is None

    def test_rejects_cobb_douglas_constraint(self):
        """A CD constraint Y = A * x1^a * x2^(1-a) is NOT CES — no inner sum, no outer Pow over Add."""
        Y, A, x1, x2, a1 = sp.symbols("Y A x1 x2 a1")
        constraints = {0: sp.Eq(Y, A * x1**a1 * x2 ** (1 - a1))}
        assert _match_ces_constraint(constraints) is None

    def test_accepts_canonical_two_input_form(self):
        """The standard CES Y = A * (alpha^(1/psi) x1^s + (1-alpha)^(1/psi) x2^s)^(1/s) must match."""
        Y, A, x1, x2, alpha, psi = sp.symbols("Y A x1 x2 alpha psi")
        s = (psi - 1) / psi
        inner = alpha ** (1 / psi) * x1**s + (1 - alpha) ** (1 / psi) * x2**s
        constraints = {7: sp.Eq(Y, A * inner ** (1 / s))}
        m = _match_ces_constraint(constraints)
        assert m is not None
        assert m["Y"] == Y
        assert m["A"] == A
        assert m["idx"] == 7
        assert sp.simplify(m["s"] - s) == 0
        inputs = dict(m["inputs"])
        assert set(inputs.keys()) == {x1, x2}

    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_accepts_arbitrary_arity(self, k):
        """Arity-agnostic: Y = A * (sum_{i=1..k} share_i * x_i^s)^(1/s) must match for any k >= 2."""
        Y, A, psi = sp.symbols("Y A psi")
        xs = sp.symbols(f"x1:{k + 1}")
        shares = sp.symbols(f"alpha1:{k + 1}")
        s = (psi - 1) / psi
        inner = sum(share ** (1 / psi) * x**s for share, x in zip(shares, xs, strict=True))
        constraints = {0: sp.Eq(Y, A * inner ** (1 / s))}
        m = _match_ces_constraint(constraints)
        assert m is not None, f"k={k}: matcher returned None"
        assert m["Y"] == Y
        assert m["A"] == A
        assert len(m["inputs"]) == k
        recovered_inputs = {x for x, _ in m["inputs"]}
        assert recovered_inputs == set(xs)

    def test_rejects_when_outer_exp_inconsistent_with_inner(self):
        """Y = A * (x1^a + x2^a)^b where a*b != 1 is not canonical CES — outer/inner exponents must reciprocate."""
        Y, A, x1, x2, a, b = sp.symbols("Y A x1 x2 a b")
        constraints = {0: sp.Eq(Y, A * (x1**a + x2**a) ** b)}
        # a*b is symbolically (a*b), not 1, so the matcher must reject.
        assert _match_ces_constraint(constraints) is None

    def test_rejects_extra_constant_factor(self):
        """Y = 2 * A * (...)^(1/s) has an extra numerical coefficient; not canonical CES."""
        Y, A, x1, x2, alpha, psi = sp.symbols("Y A x1 x2 alpha psi")
        s = (psi - 1) / psi
        inner = alpha ** (1 / psi) * x1**s + (1 - alpha) ** (1 / psi) * x2**s
        constraints = {0: sp.Eq(Y, 2 * A * inner ** (1 / s))}
        assert _match_ces_constraint(constraints) is None

    def test_detect_requires_objective(self):
        """An identity-only block must not dispatch even when the constraint matches CES shape."""
        Y, A, x1, x2, alpha, psi = sp.symbols("Y A x1 x2 alpha psi")
        s = (psi - 1) / psi
        inner = alpha ** (1 / psi) * x1**s + (1 - alpha) ** (1 / psi) * x2**s
        constraints = {0: sp.Eq(Y, A * inner ** (1 / s))}
        assert CESBlock.detect(constraints, objective=None, identities=None) is False


class TestParameterizationVariants:
    """The matcher must accept the various ``s`` and share spellings that show up across DSGE practice."""

    def test_direct_exponent_form(self):
        """Direct-exponent parameterization with simple shares.

        ``Y = A * (alpha * x1^rho + (1-alpha) * x2^rho)^(1/rho)``
        """
        Y, A, x1, x2, alpha, rho = sp.symbols("Y A x1 x2 alpha rho")
        constraints = {0: sp.Eq(Y, A * (alpha * x1**rho + (1 - alpha) * x2**rho) ** (1 / rho))}
        m = _match_ces_constraint(constraints)
        assert m is not None
        assert sp.simplify(m["s"] - rho) == 0
        shares = dict(m["inputs"])
        assert shares[x1] == alpha
        assert shares[x2] == 1 - alpha

    def test_acms_negative_exponent_form(self):
        """``Y = A * (alpha * x1^(-rho) + (1-alpha) * x2^(-rho))^(-1/rho)`` — Arrow-Chenery-Minhas-Solow original."""
        Y, A, x1, x2, alpha, rho = sp.symbols("Y A x1 x2 alpha rho")
        constraints = {0: sp.Eq(Y, A * (alpha * x1 ** (-rho) + (1 - alpha) * x2 ** (-rho)) ** (-1 / rho))}
        m = _match_ces_constraint(constraints)
        assert m is not None
        # The matcher recovers s == -rho (or some algebraic equivalent).
        assert sp.simplify(m["s"] + rho) == 0

    def test_sigma_parameterization(self):
        """``Y = A * (sum share_i^(1/sigma) * x_i^((sigma-1)/sigma))^(sigma/(sigma-1))`` — sigma instead of psi."""
        Y, A, x1, x2, alpha, sigma = sp.symbols("Y A x1 x2 alpha sigma")
        s = (sigma - 1) / sigma
        inner = alpha ** (1 / sigma) * x1**s + (1 - alpha) ** (1 / sigma) * x2**s
        constraints = {0: sp.Eq(Y, A * inner ** (1 / s))}
        m = _match_ces_constraint(constraints)
        assert m is not None
        assert sp.simplify(m["s"] - s) == 0

    def test_ratio_shares(self):
        """Shares written as ratios: ``Y = A * (alpha/(1-alpha) * x1^rho + x2^rho)^(1/rho)``."""
        Y, A, x1, x2, alpha, rho = sp.symbols("Y A x1 x2 alpha rho")
        share1 = alpha / (1 - alpha)
        constraints = {0: sp.Eq(Y, A * (share1 * x1**rho + x2**rho) ** (1 / rho))}
        m = _match_ces_constraint(constraints)
        assert m is not None
        shares = dict(m["inputs"])
        # share1 may be normalized differently by sympy; check algebraic equality.
        assert sp.simplify(shares[x1] - alpha / (1 - alpha)) == 0
        assert sp.simplify(shares[x2] - 1) == 0

    def test_two_parameter_shares(self):
        """Independent share parameters: ``Y = A * (alpha * x1^rho + beta * x2^rho)^(1/rho)``."""
        Y, A, x1, x2, alpha, beta, rho = sp.symbols("Y A x1 x2 alpha beta rho")
        constraints = {0: sp.Eq(Y, A * (alpha * x1**rho + beta * x2**rho) ** (1 / rho))}
        m = _match_ces_constraint(constraints)
        assert m is not None
        shares = dict(m["inputs"])
        assert shares[x1] == alpha
        assert shares[x2] == beta

    def test_numeric_shares(self):
        """Calibrated numeric shares: ``Y = A * (0.3 * x1^rho + 0.7 * x2^rho)^(1/rho)``."""
        Y, A, x1, x2, rho = sp.symbols("Y A x1 x2 rho")
        constraints = {0: sp.Eq(Y, A * (sp.Rational(3, 10) * x1**rho + sp.Rational(7, 10) * x2**rho) ** (1 / rho))}
        m = _match_ces_constraint(constraints)
        assert m is not None
        shares = dict(m["inputs"])
        assert shares[x1] == sp.Rational(3, 10)
        assert shares[x2] == sp.Rational(7, 10)

    def test_no_explicit_share_means_unit_share(self):
        """``Y = A * (x1^rho + x2^rho)^(1/rho)`` — implicit unit shares."""
        Y, A, x1, x2, rho = sp.symbols("Y A x1 x2 rho")
        constraints = {0: sp.Eq(Y, A * (x1**rho + x2**rho) ** (1 / rho))}
        m = _match_ces_constraint(constraints)
        assert m is not None
        shares = dict(m["inputs"])
        assert shares[x1] == 1
        assert shares[x2] == 1

    def test_mixed_share_spellings(self):
        """Mixed forms in one constraint: one input has alpha^(1/psi) share, the other has plain alpha."""
        Y, A, x1, x2, alpha, psi = sp.symbols("Y A x1 x2 alpha psi")
        s = (psi - 1) / psi
        # Asymmetric — first share has the (1/psi) Pow form, second is plain.
        inner = alpha ** (1 / psi) * x1**s + (1 - alpha) * x2**s
        constraints = {0: sp.Eq(Y, A * inner ** (1 / s))}
        m = _match_ces_constraint(constraints)
        assert m is not None
        shares = dict(m["inputs"])
        assert shares[x1] == alpha ** (1 / psi)
        assert shares[x2] == 1 - alpha

    def test_no_leading_productivity(self):
        """``Y = (alpha * x1^rho + (1-alpha) * x2^rho)^(1/rho)`` — productivity absorbed into shares or set to 1."""
        Y, x1, x2, alpha, rho = sp.symbols("Y x1 x2 alpha rho")
        constraints = {0: sp.Eq(Y, (alpha * x1**rho + (1 - alpha) * x2**rho) ** (1 / rho))}
        m = _match_ces_constraint(constraints)
        assert m is not None
        assert m["A"] is None
        assert sp.simplify(m["s"] - rho) == 0
        shares = dict(m["inputs"])
        assert shares[x1] == alpha
        assert shares[x2] == 1 - alpha

    def test_no_leading_productivity_canonical_form(self):
        """No-A in canonical (psi-1)/psi spelling: ``Y = (alpha^(1/psi) x1^s + (1-alpha)^(1/psi) x2^s)^(1/s)``."""
        Y, x1, x2, alpha, psi = sp.symbols("Y x1 x2 alpha psi")
        s = (psi - 1) / psi
        inner = alpha ** (1 / psi) * x1**s + (1 - alpha) ** (1 / psi) * x2**s
        constraints = {0: sp.Eq(Y, inner ** (1 / s))}
        m = _match_ces_constraint(constraints)
        assert m is not None
        assert m["A"] is None
