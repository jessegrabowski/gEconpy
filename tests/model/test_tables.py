import shutil
import subprocess

import arviz_stats as azs
import numpy as np
import pandas as pd
import preliz
import pytest
import xarray as xr

from gEconpy import model_from_gcn
from gEconpy.model.tables import (
    DIAGNOSTIC_FRAME_COLUMNS,
    POSTERIOR_FRAME_COLUMNS,
    PRIOR_STAT_HEADINGS,
    PRIOR_STATS,
    TABLE_GROUPS,
    TABLE_WRITERS,
    CalibrationTable,
    EquationRow,
    EquationTable,
    EstimationTable,
    ParameterRow,
    SymbolTable,
    _format_value,
)
from tests._resources.cache_compiled_models import load_and_cache_example, load_and_cache_model


@pytest.fixture(scope="module")
def rbc():
    """Built once: every test here reads the same table, and building the model dominates their runtime."""
    return load_and_cache_example("RBC")


class TestEquationTable:
    def test_a_row_carries_its_block_id_control_and_both_sides(self, rbc):
        row = next(r for r in rbc.table("equations").rows if r.equation_id == "Household.foc.C")

        assert row.block == "Household"
        assert row.label is None
        assert row.foc_control == "C_{t}"
        assert row.right == "0"
        assert r"\lambda_{t}" in row.left

    def test_a_definition_row_is_identified_by_its_position(self, rbc):
        definitions = [r for r in rbc.table("equations").rows if ".definitions." in r.equation_id]

        assert [(r.equation_id, r.left) for r in definitions] == [("Household.definitions.0", "u_{t}")]

    def test_every_equation_gets_a_row(self, rbc):
        rows = rbc.table("equations").rows

        assert [r.equation_id for r in rows if ".definitions." not in r.equation_id] == rbc._equation_ids

    @pytest.mark.parametrize(
        "example, expected",
        [
            ("RBC", ["Household", "Firm", "Technology_Shocks"]),
            ("Baxter_King_1993", ["Household", "Firm", "Fiscal_Authority"]),
        ],
        ids=["definitions_in_the_first_block", "definitions_in_a_later_block"],
    )
    def test_rows_stay_in_source_block_order(self, example, expected):
        """
        A block appears once, where the file puts it.

        RBC alone cannot pin this: its definitions are in its first block, so emitting all definitions ahead of
        all equations happens to produce the right order. Baxter-King has them in a later block.
        """
        blocks = [r.block for r in load_and_cache_example(example).table("equations").rows]

        assert list(dict.fromkeys(blocks)) == expected
        assert blocks == sorted(blocks, key=expected.index)


class TestRendering:
    @pytest.mark.parametrize(
        "example, expected",
        [
            ("RBC", ["Household", "Firm", "Technology Shocks"]),
            ("Baxter_King_1993", ["Household", "Firm", "Fiscal Authority"]),
        ],
        ids=["definitions_in_the_first_block", "definitions_in_a_later_block"],
    )
    def test_each_block_gets_exactly_one_heading(self, example, expected):
        """A repeated heading means the rows were grouped by something other than the block."""
        rendered = load_and_cache_example(example).table("equations").to_latex()

        headings = [line for line in rendered.splitlines() if line.startswith(r"\intertext")]
        assert headings == [rf"\intertext{{\textbf{{{name}}}}}" for name in expected]

    def test_a_row_keeps_the_block_identifier_the_file_spells(self, rbc):
        """The renderers turn it into prose, so the row has to keep the name a caller could filter on."""
        assert {r.block for r in rbc.table("equations").rows} == {"Household", "Firm", "Technology_Shocks"}

    def test_only_the_last_equation_line_lacks_a_terminator(self, rbc):
        r"""A stray terminator before \\end{align} is a LaTeX error, and one on a heading breaks the alignment."""
        lines = rbc.table("equations").to_latex().splitlines()[1:-1]

        assert all(not line.endswith(r"\\") for line in lines if line.startswith(r"\intertext"))
        equations = [line for line in lines if not line.startswith(r"\intertext")]
        assert all(line.endswith(r"\\") for line in equations[:-1])
        assert not equations[-1].endswith(r"\\")

    def test_a_block_heading_is_escaped(self):
        table = EquationTable([EquationRow(block="R&D Lab", equation_id=None, label=None, left="x", right="1")])

        assert r"\textbf{R\&D Lab}" in table.to_latex()

    def test_the_frame_has_one_row_per_equation_with_the_sides_joined(self, rbc):
        frame = rbc.table("equations").to_frame()

        assert list(frame.columns) == ["block", "equation_id", "label", "equation"]
        assert len(frame) == len(rbc.table("equations").rows)
        assert frame.loc[frame.equation_id == "Household.foc.C", "equation"].item().endswith(" = 0")

    def test_the_frame_and_the_latex_hold_the_same_rows(self, rbc):
        """One builder feeds both, so a row cannot appear in one rendering and not the other."""
        table = rbc.table("equations")
        equations = [line for line in table.to_latex().splitlines()[1:-1] if not line.startswith(r"\intertext")]

        assert len(equations) == len(table.rows) == len(table.to_frame())
        assert all(row.left in line for row, line in zip(table.rows, equations, strict=True))


class TestCalibrationTable:
    def test_a_row_carries_the_symbol_description_value_and_prior(self, rbc):
        row = next(r for r in rbc.table("parameters").rows if r.symbol == r"\alpha")

        assert row.description == "Capital share of output"
        assert row.value == pytest.approx(0.35)
        assert row.prior_family == "Beta"
        assert row.prior_stat("mean") == pytest.approx(0.344, abs=1e-3)

    def test_a_parameter_renders_by_the_same_rules_as_a_variable(self, tmp_path):
        r"""
        A parameter is a plain Symbol, so sympy's default printer would reach it instead.

        Every parameter in RBC renders identically either way, which is why this builds a model with two that
        do not: sympy gives ``mc`` and ``\\epsilon``.
        """
        path = tmp_path / "params.gcn"
        path.write_text(
            "symbols { Y[] { }; A[] { }; };\n"
            "block H { identities { Y[] = mc * A[] + epsilon; A[] = 1; };"
            " calibration { mc = 1.0; epsilon = 0.1; }; };\n"
        )

        symbols = {r.symbol for r in model_from_gcn(path, verbose=False).table("parameters").rows}

        assert r"\text{mc}" in symbols
        assert r"\varepsilon" in symbols

    @pytest.mark.parametrize("markup", ["latex", "typst"])
    def test_a_parameter_renders_the_same_in_its_row_and_in_an_equation(self, tmp_path, markup):
        path = tmp_path / "params.gcn"
        path.write_text(
            'symbols { phi_2 { latex = "\\kappa"; typst = "kappa"; }; };\n'
            "block H { identities { Y[] = phi_z_1 * A[] + phi_2; A[] = 1; };"
            " calibration { phi_z_1 = 0.01; phi_2 = 0.01; }; };\n"
        )
        model = model_from_gcn(path, verbose=False)

        equations = " ".join(row.left + row.right for row in model.table("equations", markup=markup).rows)
        symbols = [row.symbol for row in model.table("parameters", markup=markup).rows]

        assert len(symbols) == 2
        assert all(symbol in equations for symbol in symbols)

    def test_a_calibrated_parameter_has_no_value(self):
        """It is solved for rather than set, so printing a number would invent one."""
        model = load_and_cache_model("one_block_2_no_extra.gcn")

        calibrated = next(r for r in model.table("parameters").rows if r.symbol == r"\alpha")

        assert calibrated.value is None
        assert [r.symbol for r in model.table("parameters").rows][-1] == r"\alpha"

    def test_a_shock_standard_deviation_is_one_row_after_the_parameters(self):
        """It is estimated, so it carries a prior and no value, and it is a parameter of no equation."""
        model = load_and_cache_model("rbc_linearized.gcn")

        rows = model.table("parameters").rows
        shock_rows = [r for r in rows if r.symbol == r"\sigma_{A}"]

        assert rows[-1] == shock_rows[0]
        assert [(r.value, r.prior_family) for r in shock_rows] == [(None, "Gamma")]

    def test_an_empty_column_is_dropped(self):
        """Built here rather than from a fixture, so adding a source to a shipped model cannot silently break it."""
        table = CalibrationTable(
            [
                ParameterRow(symbol=r"\alpha", description="Capital share", value=0.35, prior=None, source=None),
                ParameterRow(symbol=r"\beta", description="Discount factor", value=0.99, prior=None, source=None),
            ]
        )

        rendered = table.to_latex()

        assert "Source" not in rendered
        assert "Prior" not in rendered
        assert "Description" in rendered

    def test_a_declared_source_brings_its_column_back(self, tmp_path):
        path = tmp_path / "sourced.gcn"
        path.write_text(
            'symbols { Y[] { }; A[] { }; alpha { name = "Capital share"; source = "Smets & Wouters (2007)"; }; };\n'
            "block H { identities { Y[] = alpha * A[]; A[] = 1; }; calibration { alpha = 0.3; }; };\n"
        )

        rendered = model_from_gcn(path, verbose=False).table("parameters").to_latex()

        assert "Source" in rendered
        assert r"Smets \& Wouters (2007)" in rendered

    def test_the_frame_keeps_every_column(self, rbc):
        """to_latex drops an empty column for printing; the frame is data and keeps it."""
        frame = rbc.table("parameters").to_frame()

        assert list(frame.columns) == [
            "symbol",
            "description",
            "value",
            "prior",
            "prior_mean",
            "prior_std",
            "prior_parameters",
            "source",
        ]
        assert len(frame) == len(rbc.table("parameters").rows)


class TestEstimationTable:
    def test_a_row_is_every_prior_in_the_parameters_table_order(self):
        """``beta`` and ``delta`` have no prior, and ``rho`` precedes ``gamma`` in the file, not the table."""
        model = load_and_cache_model("one_block_1_dist.gcn")

        symbols = [r.symbol for r in model.table("estimates").rows]

        assert symbols == [r"\alpha", r"\gamma", r"\rho", r"\sigma_{\varepsilon}"]

    def test_a_prior_prints_as_the_parameters_table_prints_it(self):
        """The two tables read one prior, so a paper printing both cannot report it two ways."""
        model = load_and_cache_model("one_block_1_dist.gcn")
        options = {"prior_stats": ["median", "std"], "include_prior_params": True}

        estimates = model.table("estimates", **options).to_frame()
        parameters = model.table("parameters", **options).to_frame().dropna(subset="prior")
        shared = [column for column in estimates.columns if column in parameters.columns]

        pd.testing.assert_frame_equal(estimates[shared], parameters[shared].reset_index(drop=True))

    def test_a_table_without_a_posterior_prints_no_posterior_column(self):
        model = load_and_cache_model("one_block_1_dist.gcn")

        header = model.write_table("estimates").splitlines()[2]

        assert header.endswith(r"\textbf{S.D.} \\")


@pytest.fixture(scope="module")
def posterior():
    """Draws for three of ``one_block_1_dist``'s four priors, and a deterministic that no prior claims."""
    rng = np.random.default_rng(0)
    means = {"alpha": 0.42, "rho": 0.93, "sigma_epsilon": 0.011, "bk_satisfied": 1.0}
    return xr.Dataset({name: (("chain", "draw"), rng.normal(mean, 0.02, (2, 100))) for name, mean in means.items()})


class TestEstimationTableWithPosterior:
    def test_a_prior_without_draws_is_dropped_and_draws_naming_no_parameter_ignored(self, posterior):
        """``gamma`` was held constant while sampling, and ``bk_satisfied`` is a deterministic of the sampler."""
        model = load_and_cache_model("one_block_1_dist.gcn")

        symbols = [r.symbol for r in model.table("estimates", idata=posterior).rows]

        assert symbols == [r"\alpha", r"\rho", r"\sigma_{\varepsilon}"]

    def test_a_parameter_sampled_without_a_declared_prior_is_a_row(self, posterior):
        """A prior can live in the PyMC model alone, as a measurement error's always does."""
        model = load_and_cache_model("one_block_1_dist.gcn")
        sampled = posterior.assign(beta=posterior["rho"], error_sigma_C=posterior["sigma_epsilon"])

        rows = {r.symbol: r for r in model.table("estimates", idata=sampled).rows}

        assert list(rows) == [r"\alpha", r"\beta", r"\rho", r"\sigma_{\varepsilon}", r"\text{error}_{\sigma,C}"]
        assert [rows[symbol].prior_stat("mean") for symbol in [r"\beta", r"\text{error}_{\sigma,C}"]] == [None, None]

    @pytest.mark.parametrize(
        "sampled_names, expected",
        [
            (["sigma_epsilon_A"], [(r"\sigma_{A}", "Gamma")]),
            (["sigma_A", "sigma_epsilon_A"], [(r"\sigma_{A}", "Gamma"), (r"\sigma_{\varepsilon,A}", "")]),
        ],
        ids=["statespace_name_only", "declared_name_sampled_too"],
    )
    def test_a_shock_standard_deviation_takes_its_declared_prior_under_the_statespace_name(
        self, posterior, sampled_names, expected
    ):
        """
        The statespace reads ``sigma_epsilon_A`` for the ``sigma_A`` the GCN file declares.

        A posterior holding ``sigma_A`` as well sampled the declared prior there, so ``sigma_epsilon_A`` is another.
        """
        model = load_and_cache_model("rbc_linearized.gcn")
        sampled = xr.Dataset({"alpha": posterior["alpha"], **dict.fromkeys(sampled_names, posterior["sigma_epsilon"])})

        rows = model.table("estimates", idata=sampled).rows

        assert [(r.symbol, r.prior_family) for r in rows][1:] == expected

    def test_a_shock_standard_deviation_checks_its_statespace_draws_against_the_declared_prior(self, posterior):
        """The draws are read under ``sigma_epsilon_A`` and compared to the ``sigma_A`` declaration."""
        model = load_and_cache_model("rbc_linearized.gcn")
        swapped = preliz.HalfNormal(sigma=0.5).rvs((2, 1000), random_state=0)
        idata = xr.DataTree.from_dict(
            {
                "posterior": xr.Dataset({"sigma_epsilon_A": posterior["sigma_epsilon"]}),
                "prior": xr.Dataset({"sigma_epsilon_A": (("chain", "draw"), swapped)}),
            }
        )

        (row,) = model.table("estimates", idata=idata).rows

        assert (row.symbol, row.prior_family) == (r"\sigma_{A}", "")
        assert row.prior_stat("mean") == pytest.approx(swapped.mean())

    def test_var_names_adds_a_variable_the_model_does_not_name(self, posterior):
        model = load_and_cache_model("one_block_1_dist.gcn")

        rows = model.table("estimates", idata=posterior, var_names=["bk_satisfied"]).rows

        assert rows[-1].posterior_mean == pytest.approx(float(posterior["bk_satisfied"].mean()))

    @pytest.mark.parametrize(
        "with_idata, options, match",
        [
            (True, {"var_names": ["shock_scale"]}, "does not have: shock_scale"),
            (False, {"var_names": ["shock_scale"]}, "need an idata"),
            (False, {"diagnostics": True}, "need an idata"),
        ],
        ids=["var_names_absent_from_posterior", "var_names_without_idata", "diagnostics_without_idata"],
    )
    def test_posterior_options_that_cannot_be_printed_raise(self, posterior, with_idata, options, match):
        model = load_and_cache_model("one_block_1_dist.gcn")
        idata = posterior if with_idata else None

        with pytest.raises(ValueError, match=match):
            model.table("estimates", idata=idata, **options)

    def test_each_row_reports_the_prior_that_was_sampled(self, posterior):
        """
        Only a prior sampled as declared prints its family and exact moments.

        ``alpha`` was sampled as declared, ``rho`` under a prior the PyMC model swapped in, and ``beta`` under one
        the GCN file never declared.
        """
        model = load_and_cache_model("one_block_1_dist.gcn")
        draws = {
            "alpha": preliz.Beta(mu=0.5, sigma=0.1).rvs((2, 1000), random_state=0),
            "rho": preliz.Normal(mu=0.8, sigma=0.05).rvs((2, 1000), random_state=1),
            "beta": preliz.Normal(mu=0.99, sigma=0.001).rvs((2, 1000), random_state=2),
        }
        prior = xr.Dataset({name: (("chain", "draw"), values) for name, values in draws.items()})
        idata = xr.DataTree.from_dict({"posterior": posterior.assign(beta=posterior["rho"]), "prior": prior})

        rows = {r.symbol: r for r in model.table("estimates", idata=idata, prior_stats=["mean", "mode"]).rows}

        assert (rows[r"\alpha"].prior_family, rows[r"\alpha"].prior_stat("mean")) == ("Beta", pytest.approx(0.5))
        for symbol, name in [(r"\rho", "rho"), (r"\beta", "beta")]:
            assert rows[symbol].prior_family == ""
            assert rows[symbol].prior_stat("mean") == pytest.approx(draws[name].mean())
            assert rows[symbol].prior_stat("mode") is None

    def test_the_posterior_columns_are_the_arviz_summary_of_the_draws(self, posterior):
        model = load_and_cache_model("one_block_1_dist.gcn")

        frame = model.table("estimates", idata=posterior, ci_kind="eti", ci_prob=0.9).to_frame()
        expected = azs.summary(
            posterior,
            var_names=["alpha", "rho", "sigma_epsilon"],
            kind="stats",
            ci_kind="eti",
            ci_prob=0.9,
            round_to="none",
        )

        np.testing.assert_allclose(frame[list(POSTERIOR_FRAME_COLUMNS)].to_numpy(), expected.to_numpy())
        assert frame[list(DIAGNOSTIC_FRAME_COLUMNS)].isna().all().all()

    def test_the_diagnostic_columns_are_the_arviz_diagnostics_of_the_draws(self, posterior):
        model = load_and_cache_model("one_block_1_dist.gcn")

        frame = model.table("estimates", idata=posterior, diagnostics=True).to_frame()
        expected = azs.summary(
            posterior, var_names=["alpha", "rho", "sigma_epsilon"], kind="diagnostics", round_to="none"
        )

        np.testing.assert_allclose(
            frame[list(DIAGNOSTIC_FRAME_COLUMNS)].to_numpy(), expected[list(DIAGNOSTIC_FRAME_COLUMNS)].to_numpy()
        )

    @pytest.mark.parametrize("diagnostics", [False, True])
    def test_the_diagnostic_columns_print_only_on_request(self, posterior, diagnostics):
        model = load_and_cache_model("one_block_1_dist.gcn")
        build_options = {"idata": posterior, "diagnostics": diagnostics}

        rendered = model.write_table("estimates", build_options=build_options)

        assert all((heading in rendered) is diagnostics for heading in ["MCSE mean", "ESS bulk", "R-hat"])

    def test_a_datatree_gives_the_table_its_posterior_group_gives(self, posterior):
        model = load_and_cache_model("one_block_1_dist.gcn")
        datatree = xr.DataTree.from_dict({"posterior": posterior})

        pd.testing.assert_frame_equal(
            model.table("estimates", idata=datatree).to_frame(), model.table("estimates", idata=posterior).to_frame()
        )

    def test_a_posterior_sharing_no_name_with_the_model_raises(self, posterior):
        """Every row dropping means the wrong sampler output, not a model with nothing estimated."""
        model = load_and_cache_model("one_block_1_dist.gcn")

        with pytest.raises(ValueError, match="shares no variable with the model's parameters"):
            model.table("estimates", idata=posterior[["bk_satisfied"]])

    def test_sampler_output_without_a_posterior_group_raises(self, posterior):
        """Prior predictive output has only a ``prior`` group, and xarray's own error names no group at all."""
        model = load_and_cache_model("one_block_1_dist.gcn")
        prior_only = xr.DataTree.from_dict({"prior": posterior})

        with pytest.raises(ValueError, match="no posterior group to summarize\\. Its groups are: prior"):
            model.table("estimates", idata=prior_only)

    def test_a_posterior_variable_with_extra_dimensions_raises(self, posterior):
        model = load_and_cache_model("one_block_1_dist.gcn")
        vector = posterior.assign(alpha=posterior["alpha"].expand_dims(lag=2, axis=-1))

        with pytest.raises(ValueError, match="dimensions beyond chain and draw: alpha"):
            model.table("estimates", idata=vector)

    @pytest.mark.parametrize(
        "writer, heading, interval_opening",
        [("latex", r"\textbf{90\% ETI}", "& ["), ("typst", "[*90% ETI*]", "[\\[")],
        ids=["latex", "typst"],
    )
    def test_the_interval_heading_and_cell_survive_the_writer(self, posterior, writer, heading, interval_opening):
        """A bare ``%`` starts a LaTeX comment, and a bare ``[`` opens Typst content, so both are escaped."""
        model = load_and_cache_model("one_block_1_dist.gcn")
        build_options = {"idata": posterior, "ci_kind": "eti", "ci_prob": 0.9}

        rendered = model.write_table("estimates", writer=writer, build_options=build_options)

        assert heading in rendered
        assert rendered.count(interval_opening) == 3


class TestEmptyTable:
    """An empty table must not emit a tabular with no columns or an align with no rows; pdflatex rejects both."""

    @pytest.mark.parametrize(
        "table",
        [EquationTable([]), CalibrationTable([]), EstimationTable([])],
        ids=["equations", "calibration", "estimates"],
    )
    def test_an_empty_table_renders_as_nothing(self, table):
        assert table.to_latex() == ""

    @pytest.mark.parametrize("style", [{}, {"caption": "Shocks"}, {"label": "tab:s"}, {"size": "small"}])
    def test_an_empty_table_stays_empty_under_any_style(self, style):
        """A caption on no table still floats, numbering a phantom entry in the list of tables."""
        assert SymbolTable([]).to_latex(**style) == ""

    @pytest.mark.parametrize(
        "table, columns",
        [
            (EquationTable([]), ["block", "equation_id", "label", "equation"]),
            (
                CalibrationTable([]),
                ["symbol", "description", "value", "prior", "prior_mean", "prior_std", "prior_parameters", "source"],
            ),
            (
                EstimationTable([]),
                [
                    "symbol",
                    "description",
                    "prior",
                    "prior_mean",
                    "prior_std",
                    "prior_parameters",
                    *POSTERIOR_FRAME_COLUMNS,
                    *DIAGNOSTIC_FRAME_COLUMNS,
                ],
            ),
        ],
        ids=["equations", "calibration", "estimates"],
    )
    def test_an_empty_frame_keeps_its_columns(self, table, columns):
        """Returning a frame with no columns breaks the contract the docstring states."""
        assert list(table.to_frame().columns) == columns


class TestValueFormatting:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.35, "0.35"),
            (2.0, "2"),
            (1e-07, r"$1 \times 10^{-7}$"),
            (1.23457e08, r"$1.23457 \times 10^{8}$"),
            (None, ""),
        ],
        ids=["decimal", "integral", "small", "large", "calibrated"],
    )
    def test_a_value_never_prints_bare_scientific_notation(self, value, expected):
        """``1e-07`` in a tabular cell is outside math mode and prints literally."""
        assert _format_value(value) == expected


def test_the_rendered_calibration_has_the_expected_shape():
    """Pins column order, the hline structure and the escaping, which field assertions leave unchecked.

    The rows are written out rather than taken from a model, because a prior's parameters come from a
    moment-matching solve and their third significant figure moves with the scipy build.
    """
    table = CalibrationTable(
        rows=[
            ParameterRow(
                symbol=r"\alpha",
                description="Capital share of output",
                value=0.35,
                prior=preliz.Beta(alpha=21.8, beta=41.6),
                source="Smets & Wouters (2007)",
            ),
            ParameterRow(symbol=r"\delta", description="Depreciation rate", value=0.02, prior=None, source=None),
        ]
    )

    assert table.to_latex() == (
        "\\begin{tabular}{lp{0.3\\linewidth}llrrp{0.2\\linewidth}}\n"
        "\\hline\n"
        "\\textbf{Parameter} & \\textbf{Description} & \\textbf{Value} & \\textbf{Prior} & "
        "\\textbf{Mean} & \\textbf{S.D.} & \\textbf{Source} \\\\\n"
        "\\hline\n"
        "$\\alpha$ & Capital share of output & 0.35 & Beta & 0.344 & 0.0592 & Smets \\& Wouters (2007) \\\\\n"
        "$\\delta$ & Depreciation rate & 0.02 &  &  &  &  \\\\\n"
        "\\hline\n"
        "\\end{tabular}"
    )


@pytest.mark.parametrize("group", ["equations", "parameters", "estimates"])
def test_the_rendered_table_compiles(rbc, group, tmp_path):
    r"""
    The feature's whole contract is valid LaTeX, and an empty table once produced output pdflatex rejected.

    ``amssymb`` is needed as well as ``amsmath``: the expectation operator renders as ``\\mathbb{E}``.
    """
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        pytest.skip("pdflatex not installed")

    body = rbc.table(group).to_latex()
    source = tmp_path / "table.tex"
    source.write_text(
        f"\\documentclass{{article}}\n\\usepackage{{amsmath,amssymb}}\n\\begin{{document}}\n{body}\n\\end{{document}}\n"
    )

    result = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", source.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout[-2000:]


@pytest.mark.parametrize(
    "group, columns",
    [
        ("equations", ["block", "equation_id", "label", "equation"]),
        ("variables", ["symbol", "description"]),
        ("shocks", ["symbol", "description"]),
        (
            "parameters",
            ["symbol", "description", "value", "prior", "prior_mean", "prior_std", "prior_parameters", "source"],
        ),
    ],
)
def test_every_group_declares_the_columns_its_frame_carries(rbc, group, columns):
    table = rbc.table(group)

    assert list(table.to_frame().columns) == columns
    assert len(table.to_frame()) == len(table.rows)


@pytest.mark.parametrize("group", ["variables", "shocks"])
def test_a_symbol_group_covers_every_symbol(rbc, group):
    """A row builder that filtered or deduplicated would leave a symbol out of the paper."""
    assert len(rbc.table(group).rows) == len(getattr(rbc, group))


def test_the_equations_group_covers_every_equation(rbc):
    """Definitions are extra rows; every solved equation must still appear, in order."""
    ids = [row.equation_id for row in rbc.table("equations").rows if ".definitions." not in row.equation_id]

    assert ids == rbc._equation_ids


def test_an_unknown_group_raises(rbc):
    """A typo in a group name should fail loudly rather than return an empty table."""
    with pytest.raises(ValueError, match="group must be one of equations, variables, shocks, parameters"):
        rbc.table("parameter")


def test_a_symbol_row_carries_its_declared_caption_and_latex(rbc):
    """The variables group reads the same declarations the notebook repr does."""
    frame = rbc.table("variables").to_frame().set_index("symbol")

    assert frame.loc[r"\lambda_{t}", "description"] == "Marginal utility of wealth"
    assert frame.loc["K_{t}", "description"] == "Capital stock"


def test_a_shock_row_renders_through_the_latex_printer(rbc):
    r"""``epsilon`` renders as ``\\varepsilon``, which a bare sympy printer would not do."""
    symbols = rbc.table("shocks").to_frame()["symbol"].tolist()

    assert symbols == [r"\varepsilon_{A,t}"]


def test_a_symbol_table_drops_the_description_column_when_nothing_declares_one(tmp_path):
    """A model with no declared names should not print an empty caption column."""
    path = tmp_path / "bare.gcn"
    path.write_text(
        "symbols { Y[] { }; A[] { }; alpha { }; };\n"
        "block H { identities { Y[] = alpha * A[]; A[] = 1; }; calibration { alpha = 0.3; }; };\n"
    )

    rendered = model_from_gcn(path, verbose=False).table("variables").to_latex()

    assert "Description" not in rendered
    assert "Symbol" in rendered


def test_a_declared_name_replaces_the_whole_variable_name(tmp_path):
    path = tmp_path / "declared.gcn"
    path.write_text(
        'symbols { p_H_tilde[] { latex = "\\hat{p}_H"; }; };\n'
        "block H { identities { log(p_H_tilde[]) = 0.9 * log(p_H_tilde[-1]); }; };\n"
    )

    rows = model_from_gcn(path, verbose=False).table("variables").rows

    assert [row.symbol for row in rows] == [r"\hat{p}_{H,t}"]


class TestWriteTable:
    def test_an_unknown_writer_raises(self, rbc):
        with pytest.raises(ValueError, match="writer must be one of latex, typst"):
            rbc.write_table("parameters", writer="markdown")

    def test_a_caption_and_label_wrap_the_table(self, rbc):
        """Neither can be referenced outside a table environment, so asking for one implies the wrapper."""
        rendered = rbc.write_table("parameters", caption="Calibrated parameters", label="tab:calib")

        assert r"\begin{table}[htbp]" in rendered
        assert r"\caption{Calibrated parameters}" in rendered
        assert r"\label{tab:calib}" in rendered

    def test_a_caption_carries_math_unescaped(self, rbc):
        """A table caption in this literature routinely carries math, so it is the author's own LaTeX."""
        rendered = rbc.write_table("parameters", caption=r"Calibration of $\beta$")

        assert r"\caption{Calibration of $\beta$}" in rendered

    def test_no_style_leaves_a_bare_tabular(self, rbc):
        rendered = rbc.write_table("parameters")

        assert r"\begin{table}" not in rendered
        assert rendered.startswith(r"\begin{tabular}")

    def test_booktabs_replaces_the_hlines(self, rbc):
        rendered = rbc.write_table("variables", booktabs=True)

        assert r"\toprule" in rendered
        assert r"\midrule" in rendered
        assert r"\bottomrule" in rendered
        assert r"\hline" not in rendered

    def test_size_is_scoped_to_the_table(self, rbc):
        """A size command is a declaration, so unbraced it shrinks every paragraph after the table."""
        rendered = rbc.write_table("variables", size="footnotesize")

        assert rendered.startswith("{\\footnotesize\n")
        assert rendered.endswith("}")

    def test_widths_replace_the_column_spec(self, rbc):
        rendered = rbc.write_table("variables", widths=["c", r"p{0.9\linewidth}"])

        assert r"\begin{tabular}{cp{0.9\linewidth}}" in rendered

    def test_widths_of_the_wrong_length_raise(self, rbc):
        """A dropped empty column changes how many specifiers are needed, so the count is checked."""
        with pytest.raises(ValueError, match="one specifier per printed column"):
            rbc.write_table("variables", widths=["c"])

    def test_a_path_receives_the_markup(self, rbc, tmp_path):
        out = tmp_path / "table.tex"

        returned = rbc.write_table("shocks", path=out)

        assert out.read_text() == returned

    def test_a_fully_styled_table_compiles(self, rbc, tmp_path):
        """Every style option at once is the combination most likely to produce invalid LaTeX."""
        pdflatex = shutil.which("pdflatex")
        if pdflatex is None:
            pytest.skip("pdflatex not installed")

        body = rbc.write_table(
            "parameters",
            caption="Calibration",
            label="tab:calib",
            size="small",
            booktabs=True,
        )
        source = tmp_path / "styled.tex"
        source.write_text(
            "\\documentclass{article}\n\\usepackage{amsmath,amssymb,booktabs}\n"
            f"\\begin{{document}}\n{body}\n\\end{{document}}\n"
        )

        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", source.name],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stdout[-2000:]


@pytest.mark.parametrize("group", TABLE_GROUPS)
@pytest.mark.parametrize("writer", list(TABLE_WRITERS))
def test_every_announced_writer_renders_every_group(rbc, writer, group):
    """A writer wired onto one table class but not the others passes a single-group test and fails in use."""
    assert rbc.write_table(group, writer=writer)


def test_a_style_option_the_group_does_not_take_raises(rbc):
    """Equations render as an align environment, which has no rules to style, so booktabs must not be ignored."""
    with pytest.raises(TypeError, match="'equations' takes no style option 'booktabs'"):
        rbc.write_table("equations", booktabs=True)


def test_a_build_option_the_group_does_not_take_raises(rbc):
    """``expectations`` means nothing outside the equations group, and silence would hide a flag going nowhere."""
    with pytest.raises(TypeError, match="'variables' takes no build option 'expectations'"):
        rbc.table("variables", expectations=False)


def test_an_unknown_style_option_names_the_group_not_the_renderer(rbc):
    """Python's own message names the bound method, which is private, so a typo points at the wrong thing."""
    with pytest.raises(TypeError, match="Accepted: caption, label, size, widths, booktabs"):
        rbc.write_table("parameters", captoin="Calibration")


def test_an_empty_table_rejects_widths():
    """An empty table prints no columns, so any width count is wrong, and silence would hide a bad script."""
    with pytest.raises(ValueError, match="one specifier per printed column"):
        SymbolTable([]).to_latex(widths=["c", "c"])


@pytest.mark.parametrize(
    "options, headings",
    [
        ({}, ["Prior", "Mean", "S.D."]),
        ({"prior_stats": ["median", "skewness", "kurtosis"]}, ["Prior", "Median", "Skew", "Kurtosis"]),
        ({"prior_stats": [], "include_prior_params": True}, ["Prior", "Parameters"]),
        ({"include_prior_params": True}, ["Prior", "Mean", "S.D.", "Parameters"]),
    ],
    ids=["default", "other_stats", "params_only", "stats_and_params"],
)
def test_the_prior_presentation_is_chosen_by_the_caller(rbc, options, headings):
    """A prior as one string is the least readable column in the table, so the caller picks its columns."""
    rendered = rbc.table("parameters", **options).to_latex()
    header = next(line for line in rendered.splitlines() if "textbf{Parameter}" in line)
    printed = [
        h for h in ["Prior", "Mean", "S.D.", "Median", "Skew", "Kurtosis", "Parameters"] if f"\\textbf{{{h}}}" in header
    ]

    assert printed == headings


def test_an_unknown_prior_statistic_raises(rbc):
    """A typo must not produce a silently empty column in a table bound for a paper."""
    with pytest.raises(ValueError, match="got medain"):
        rbc.table("parameters", prior_stats=["medain"])


def test_an_unknown_statistic_raises_even_when_no_parameter_has_a_prior():
    """Validating against the rows would pass a typo on a model whose parameters are all calibrated."""
    with pytest.raises(ValueError, match="got medain"):
        CalibrationTable(rows=[ParameterRow("x", None, 1.0, None, None)], prior_stats=["medain"])


def test_a_distribution_method_that_is_not_a_statistic_is_rejected():
    """``rvs`` and ``plot_pdf`` are on the base class too, so presence alone is not the test."""
    with pytest.raises(ValueError, match="got rvs"):
        CalibrationTable(rows=[], prior_stats=["rvs"])


@pytest.mark.parametrize("name", PRIOR_STATS)
def test_every_announced_statistic_renders(rbc, name):
    """A name in PRIOR_STATS that some family cannot compute would raise only for that family's priors."""
    heading = PRIOR_STAT_HEADINGS.get(name, name.title())

    frame = rbc.table("parameters", prior_stats=[name]).to_frame()
    rendered = rbc.table("parameters", prior_stats=[name]).to_latex()

    assert rf"\textbf{{{heading}}}" in rendered
    assert frame[f"prior_{name}"].notna().all()


def test_a_prior_prints_its_moments_to_three_figures(rbc):
    """``%g`` gives six, and a prior mean of 0.343849 is noise in a paper."""
    rendered = rbc.table("parameters").to_latex()

    assert "& 0.344 &" in rendered
    assert "0.343849" not in rendered


def test_a_parameter_with_no_prior_reports_nothing_about_one():
    """A calibrated parameter has a value and no distribution, and must not print a family or a moment."""
    row = ParameterRow(symbol=r"\delta", description="Depreciation rate", value=0.02, prior=None, source=None)

    assert row.prior_family == ""
    assert row.prior_stat("mean") is None
    assert row.prior_parameters == ""


def test_a_table_of_calibrated_parameters_drops_every_prior_column():
    """The empty-column rule should take the family and the moments out together, not leave blank headings."""
    table = CalibrationTable(
        rows=[ParameterRow(symbol=r"\delta", description="Depreciation rate", value=0.02, prior=None, source=None)],
        include_prior_params=True,
    )

    rendered = table.to_latex()

    assert not [h for h in ["Prior", "Mean", "S.D.", "Parameters"] if f"\\textbf{{{h}}}" in rendered]
    assert r"\textbf{Value}" in rendered


def test_prior_parameters_rounds_like_the_moment_columns_beside_it():
    """The raw fitted values run to six figures, which is the noise this column exists to avoid."""
    row = ParameterRow(
        symbol=r"\alpha", description=None, value=None, prior=preliz.Beta(alpha=21.8346, beta=41.6247), source=None
    )

    assert row.prior_parameters == "alpha=21.8, beta=41.6"
