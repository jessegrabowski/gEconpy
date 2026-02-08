import pytest

from gEconpy.parser.ast import (
    T_MINUS_1,
    GCNDistribution,
    GCNEquation,
    T,
    Variable,
)
from gEconpy.parser.grammar.blocks import (
    parse_block,
    parse_block_from_text,
)


class TestParseBlock:
    def test_empty_block(self):
        block = parse_block("TEST", "")
        assert block.name == "TEST"
        assert block.definitions == []
        assert block.controls == []
        assert block.objective == []

    def test_identities_only(self):
        content = """
        identities
        {
            Y[] = C[] + I[];
            K[] = (1 - delta) * K[-1] + I[];
        };
        """
        block = parse_block("EQUILIBRIUM", content)
        assert block.name == "EQUILIBRIUM"
        assert len(block.identities) == 2
        assert all(isinstance(eq, GCNEquation) for eq in block.identities)

    def test_controls_parsing(self):
        content = """
        controls
        {
            C[], L[], I[], K[];
        };
        """
        block = parse_block("HOUSEHOLD", content)
        assert len(block.controls) == 4
        names = [v.name for v in block.controls]
        assert names == ["C", "L", "I", "K"]

    def test_shocks_parsing(self):
        content = """
        shocks
        {
            epsilon_A[], epsilon_B[];
        };
        """
        block = parse_block("SHOCKS", content)
        assert len(block.shocks) == 2
        names = [v.name for v in block.shocks]
        assert "epsilon_A" in names
        assert "epsilon_B" in names

    def test_definitions_parsing(self):
        content = """
        definitions
        {
            u[] = log(C[]) - Theta * L[];
        };
        """
        block = parse_block("HOUSEHOLD", content)
        assert len(block.definitions) == 1
        assert block.definitions[0].lhs == Variable(name="u")

    def test_objective_parsing(self):
        content = """
        objective
        {
            U[] = u[] + beta * E[][U[1]];
        };
        """
        block = parse_block("HOUSEHOLD", content)
        assert block.objective[0].lhs == Variable(name="U")

    def test_constraints_with_lagrange(self):
        content = """
        constraints
        {
            C[] + I[] = r[] * K[-1] + w[] * L[] : lambda[];
            K[] = (1 - delta) * K[-1] + I[] : q[];
        };
        """
        block = parse_block("HOUSEHOLD", content)
        assert len(block.constraints) == 2
        assert block.constraints[0].lagrange_multiplier == "lambda"
        assert block.constraints[1].lagrange_multiplier == "q"

    def test_calibration_with_equations(self):
        content = """
        calibration
        {
            beta = 0.99;
            delta = 0.025;
            alpha = 0.35;
        };
        """
        block = parse_block("HOUSEHOLD", content)
        assert len(block.calibration) == 3
        assert all(isinstance(c, GCNEquation) for c in block.calibration)

    def test_calibration_with_distributions(self):
        content = """
        calibration
        {
            beta ~ Beta(alpha=2, beta=5) = 0.99;
            alpha ~ maxent(Beta(), lower=0.2, upper=0.5) = 0.35;
        };
        """
        block = parse_block("HOUSEHOLD", content)
        assert len(block.calibration) == 2
        assert all(isinstance(c, GCNDistribution) for c in block.calibration)
        assert block.calibration[0].parameter_name == "beta"
        assert block.calibration[1].wrapper_name == "maxent"

    def test_calibration_mixed(self):
        content = """
        calibration
        {
            beta ~ Beta(alpha=2, beta=5) = 0.99;
            delta = 0.025;
        };
        """
        block = parse_block("HOUSEHOLD", content)
        assert len(block.calibration) == 2
        assert isinstance(block.calibration[0], GCNDistribution)
        assert isinstance(block.calibration[1], GCNEquation)


class TestFullBlock:
    def test_household_block(self):
        content = """
        definitions
        {
            u[] = C[] ^ (1 - sigma_C) / (1 - sigma_C) - L[] ^ (1 + sigma_L) / (1 + sigma_L);
        };

        controls
        {
            C[], L[], I[], K[];
        };

        objective
        {
            U[] = u[] + beta * E[][U[1]];
        };

        constraints
        {
            C[] + I[] = r[] * K[-1] + w[] * L[] : lambda[];
            K[] = (1 - delta) * K[-1] + I[];
        };

        calibration
        {
            beta = 0.99;
            delta = 0.025;
            sigma_C = 1.5;
            sigma_L = 2.0;
        };
        """
        block = parse_block("HOUSEHOLD", content)

        assert block.name == "HOUSEHOLD"
        assert len(block.definitions) == 1
        assert len(block.controls) == 4
        assert block.objective is not None
        assert len(block.constraints) == 2
        assert len(block.calibration) == 4
        assert block.has_optimization_problem()

    def test_firm_block(self):
        content = """
        controls
        {
            K[-1], L[];
        };

        objective
        {
            TC[] = -(r[] * K[-1] + w[] * L[]);
        };

        constraints
        {
            Y[] = A[] * K[-1] ^ alpha * L[] ^ (1 - alpha) : mc[];
        };

        calibration
        {
            alpha = 0.35;
        };
        """
        block = parse_block("FIRM", content)

        assert block.name == "FIRM"
        assert len(block.controls) == 2
        # K[-1] should have time index -1
        k_control = next(c for c in block.controls if c.name == "K")
        assert k_control.time_index == T_MINUS_1

    def test_technology_shocks_block(self):
        content = """
        identities
        {
            log(A[]) = rho_A * log(A[-1]) + epsilon_A[];
        };

        shocks
        {
            epsilon_A[];
        };

        calibration
        {
            rho_A = 0.95;
        };
        """
        block = parse_block("TECHNOLOGY_SHOCKS", content)

        assert len(block.identities) == 1
        assert len(block.shocks) == 1
        assert block.shocks[0].name == "epsilon_A"

    def test_shock_with_distribution(self):
        content = """
        shocks
        {
            epsilon[] ~ Normal(mu=0, sigma=sigma_epsilon);
        };
        """
        block = parse_block("TEST", content)

        assert len(block.shocks) == 1
        assert block.shocks[0].name == "epsilon"
        assert len(block.shock_distributions) == 1
        dist = block.shock_distributions[0]
        assert dist.parameter_name == "epsilon"
        assert dist.dist_name == "Normal"
        assert dist.dist_kwargs["mu"] == 0
        assert dist.dist_kwargs["sigma"] == "sigma_epsilon"

    def test_mixed_shocks_plain_and_distribution(self):
        content = """
        shocks
        {
            epsilon_A[], epsilon_B[];
            epsilon_C[] ~ Normal(mu=0, sigma=0.01);
        };
        """
        block = parse_block("TEST", content)

        assert len(block.shocks) == 3
        shock_names = {s.name for s in block.shocks}
        assert shock_names == {"epsilon_A", "epsilon_B", "epsilon_C"}
        assert len(block.shock_distributions) == 1
        assert block.shock_distributions[0].parameter_name == "epsilon_C"


class TestParseBlockFromText:
    def test_simple_block(self):
        text = """
        block EQUILIBRIUM
        {
            identities
            {
                B[] = 0;
            };
        };
        """
        block = parse_block_from_text(text)
        assert block.name == "EQUILIBRIUM"
        assert len(block.identities) == 1

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid block format"):
            parse_block_from_text("not a block")

    def test_case_insensitive_block_keyword(self):
        text = """
        BLOCK TEST
        {
            identities { Y[] = C[]; };
        };
        """
        block = parse_block_from_text(text)
        assert block.name == "TEST"


class TestEdgeCases:
    def test_multiline_equation(self):
        content = """
        identities
        {
            Y[ss] = (r[ss] / (r[ss] - delta * alpha)) ^ (sigma_C / (sigma_C + sigma_L)) *
                (w[ss] * (w[ss] / (1 - alpha)) ^ sigma_L) ^ (1 / (sigma_C + sigma_L));
        };
        """
        block = parse_block("STEADY_STATE", content)
        assert len(block.identities) == 1

    def test_empty_components(self):
        content = """
        controls
        {
        };
        identities
        {
        };
        """
        block = parse_block("TEST", content)
        assert block.controls == []
        assert block.identities == []

    def test_whitespace_variations(self):
        content = """
        controls{C[],L[],K[];};
        """
        block = parse_block("TEST", content)
        assert len(block.controls) == 3


class TestRealGCNBlocks:
    """Test parsing of blocks extracted from real GCN files."""

    def test_one_block_household(self):
        """Parse a HOUSEHOLD block similar to one_block_1.gcn."""
        content = """
        definitions
        {
            u[] = (C[] ^ (1 - gamma) - 1) / (1 - gamma);
        };

        controls
        {
            C[], K[];
        };

        objective
        {
            U[] = u[] + beta * E[][U[1]];
        };

        constraints
        {
            C[] + K[] - (1 - delta) * K[-1] = A[] * K[-1] ^ alpha : lambda[];
        };

        identities
        {
            log(A[]) = rho * log(A[-1]) + epsilon[];
        };

        shocks
        {
            epsilon[];
        };

        calibration
        {
            alpha = 0.4;
            beta = 0.99;
            delta = 0.02;
            rho = 0.95;
            gamma = 1.5;
        };
        """
        block = parse_block("HOUSEHOLD", content)

        assert len(block.definitions) == 1
        assert len(block.controls) == 2
        assert block.objective is not None
        assert len(block.constraints) == 1
        assert block.constraints[0].lagrange_multiplier == "lambda"
        assert len(block.identities) == 1
        assert len(block.shocks) == 1
        assert block.shocks[0].name == "epsilon"
        assert len(block.calibration) == 5
        assert block.has_optimization_problem()

    def test_nk_wage_setting_block(self):
        """Parse a WAGE_SETTING block similar to full_nk.gcn."""
        content = """
        definitions
        {
            L_d_star[] = (w[] / w_star[]) ^ ((1 + psi_w) / psi_w) * L[];
        };

        identities
        {
            LHS_w[] = RHS_w[];
            LHS_w[] = 1 / (1 + psi_w) * w_star[] * lambda[] * L_d_star[] +
                beta * eta_w * E[][pi[1] * (w_star[1] / w_star[]) ^ (1 / psi_w) * LHS_w[1]];
        };

        calibration
        {
            psi_w = 0.782;
            eta_w = 0.75;
        };
        """
        block = parse_block("WAGE_SETTING", content)

        assert len(block.definitions) == 1
        assert len(block.identities) == 2
        assert len(block.calibration) == 2
        assert not block.has_optimization_problem()

    def test_nk_monetary_policy_block(self):
        """Parse a MONETARY_POLICY block with shocks."""
        content = """
        identities
        {
            log(r_G[] / r_G[ss]) = gamma_R * log(r_G[-1] / r_G[ss]) +
                (1 - gamma_R) * gamma_pi * log(pi[] / pi[ss]) + epsilon_R[];
        };

        shocks
        {
            epsilon_R[], epsilon_pi[];
        };

        calibration
        {
            gamma_R = 0.9;
            gamma_pi = 1.5;
        };
        """
        block = parse_block("MONETARY_POLICY", content)

        assert len(block.identities) == 1
        assert len(block.shocks) == 2
        shock_names = [s.name for s in block.shocks]
        assert "epsilon_R" in shock_names
        assert "epsilon_pi" in shock_names

    def test_block_with_distributions(self):
        """Parse calibration with prior distributions like in RBC.gcn."""
        content = """
        calibration
        {
            beta ~ maxent(Beta(), lower=0.95, upper=0.999, mass=0.99) = 0.99;
            delta ~ maxent(Beta(), lower=0.01, upper=0.05, mass=0.99) = 0.02;
            sigma_C ~ maxent(Gamma(), lower=1.01, upper=10.0, mass=0.99) = 1.5;
            sigma_L ~ maxent(Gamma(), lower=1.000, upper=10.0, mass=0.99) = 2.0;
        };
        """
        block = parse_block("HOUSEHOLD", content)

        assert len(block.calibration) == 4
        assert all(isinstance(c, GCNDistribution) for c in block.calibration)

        param_names = [c.parameter_name for c in block.calibration]
        assert param_names == ["beta", "delta", "sigma_C", "sigma_L"]

    def test_open_rbc_household(self):
        """Parse a block with complex definitions from open_rbc.gcn."""
        content = """
        definitions
        {
            u[] = 1/(1-gamma)*((C[] - 1 / omega * N[] ^ omega) ^ (1 - gamma) - 1);
            I[] = K[] - (1 - delta) * K[-1];
            Cadjcost[] = psi/2*(K[] - K[-1])^2;
            Y[] = A[] * K[-1] ^ alpha * N[] ^ (1 - alpha);
        };

        controls
        {
            C[], N[], K[], IIP[];
        };

        objective
        {
            U[] = u[] + beta * E[][U[1]];
        };

        constraints
        {
            C[] + I[] + Cadjcost[] + IIP[] = Y[] + (1+r_given[-1])*IIP[-1] : lambda[];
        };
        """
        block = parse_block("HOUSEHOLD", content)

        assert len(block.definitions) == 4
        assert len(block.controls) == 4
        assert block.objective is not None
        assert len(block.constraints) == 1

    def test_steady_state_block(self):
        """Parse a STEADY_STATE block with ss time indices."""
        content = """
        identities
        {
            A[ss] = 1;
            r[ss] = 1 / beta - (1 - delta);
            K[ss] = alpha * Y[ss] / r[ss];
            L[ss] = (1 - alpha) * Y[ss] / w[ss];
        };
        """
        block = parse_block("STEADY_STATE", content)

        assert len(block.identities) == 4
        # All LHS should be steady state variables
        for eq in block.identities:
            assert eq.lhs.time_index.is_steady_state
