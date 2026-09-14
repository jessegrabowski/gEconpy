gEconpy
=======

.. currentmodule:: gEconpy

.. rubric:: Re-exported

.. autosummary::

    dynare_convert.make_mod_file
    model.build.model_from_gcn
    model.build.statespace_from_gcn
    model.perfect_foresight.solve.solve_perfect_foresight
    model.sampling.bounds_from_priors
    model.sampling.sample_from_priors
    model.sampling.sample_from_priors_qmc
    model.sampling.sample_uniform
    model.sampling.sample_uniform_from_priors
    model.simulate.impulse_response_function
    model.simulate.simulate
    model.statespace.data_from_prior
    model.statespace.prepare_mixed_frequency_data
    model.statistics.covariance.autocorrelation_matrix
    model.statistics.covariance.autocovariance_matrix
    model.statistics.covariance.build_Q_matrix
    model.statistics.covariance.stationary_covariance_matrix
    model.statistics.formatting.matrix_to_dataframe
    model.statistics.perturbation_diagnostics.check_bk_condition
    model.statistics.perturbation_diagnostics.prior_solvability_check
    model.statistics.perturbation_diagnostics.solvability_check
    model.statistics.perturbation_diagnostics.summarize_perturbation_solution
    model.statistics.validation.check_steady_state
    model.steady_state.print_steady_state
    parser.html.print_gcn_file

.. rubric:: Submodules

.. toctree::
    :maxdepth: 1

    gEconpy.classes.containers <gEconpy.classes.containers>
    gEconpy.classes.distributions <gEconpy.classes.distributions>
    gEconpy.classes.time_aware_symbol <gEconpy.classes.time_aware_symbol>
    gEconpy.data <gEconpy.data>
    gEconpy.data.examples <gEconpy.data.examples>
    gEconpy.dynare_convert <gEconpy.dynare_convert>
    gEconpy.exceptions <gEconpy.exceptions>
    gEconpy.model.block <gEconpy.model.block>
    gEconpy.model.block.basic <gEconpy.model.block.basic>
    gEconpy.model.block.ces <gEconpy.model.block.ces>
    gEconpy.model.block.cobb_douglas <gEconpy.model.block.cobb_douglas>
    gEconpy.model.block.registry <gEconpy.model.block.registry>
    gEconpy.model.build <gEconpy.model.build>
    gEconpy.model.compile <gEconpy.model.compile>
    gEconpy.model.model <gEconpy.model.model>
    gEconpy.model.parameters <gEconpy.model.parameters>
    gEconpy.model.perfect_foresight <gEconpy.model.perfect_foresight>
    gEconpy.model.perfect_foresight.assemble <gEconpy.model.perfect_foresight.assemble>
    gEconpy.model.perfect_foresight.compile <gEconpy.model.perfect_foresight.compile>
    gEconpy.model.perfect_foresight.solve <gEconpy.model.perfect_foresight.solve>
    gEconpy.model.perfect_foresight.validation <gEconpy.model.perfect_foresight.validation>
    gEconpy.model.perturbation <gEconpy.model.perturbation>
    gEconpy.model.sampling <gEconpy.model.sampling>
    gEconpy.model.simplification <gEconpy.model.simplification>
    gEconpy.model.simulate <gEconpy.model.simulate>
    gEconpy.model.statespace <gEconpy.model.statespace>
    gEconpy.model.statistics <gEconpy.model.statistics>
    gEconpy.model.statistics.covariance <gEconpy.model.statistics.covariance>
    gEconpy.model.statistics.formatting <gEconpy.model.statistics.formatting>
    gEconpy.model.statistics.perturbation_diagnostics <gEconpy.model.statistics.perturbation_diagnostics>
    gEconpy.model.statistics.validation <gEconpy.model.statistics.validation>
    gEconpy.model.steady_state <gEconpy.model.steady_state>
    gEconpy.model.timing <gEconpy.model.timing>
    gEconpy.parser <gEconpy.parser>
    gEconpy.parser.ast <gEconpy.parser.ast>
    gEconpy.parser.ast.nodes <gEconpy.parser.ast.nodes>
    gEconpy.parser.ast.printer <gEconpy.parser.ast.printer>
    gEconpy.parser.ast.validation <gEconpy.parser.ast.validation>
    gEconpy.parser.ast.visitor <gEconpy.parser.ast.visitor>
    gEconpy.parser.error_catalog <gEconpy.parser.error_catalog>
    gEconpy.parser.errors <gEconpy.parser.errors>
    gEconpy.parser.formatting <gEconpy.parser.formatting>
    gEconpy.parser.grammar <gEconpy.parser.grammar>
    gEconpy.parser.grammar.blocks <gEconpy.parser.grammar.blocks>
    gEconpy.parser.grammar.expressions <gEconpy.parser.grammar.expressions>
    gEconpy.parser.grammar.gcn_file <gEconpy.parser.grammar.gcn_file>
    gEconpy.parser.grammar.special_blocks <gEconpy.parser.grammar.special_blocks>
    gEconpy.parser.grammar.statements <gEconpy.parser.grammar.statements>
    gEconpy.parser.html <gEconpy.parser.html>
    gEconpy.parser.loader <gEconpy.parser.loader>
    gEconpy.parser.preprocessor <gEconpy.parser.preprocessor>
    gEconpy.parser.suggestions <gEconpy.parser.suggestions>
    gEconpy.parser.transform <gEconpy.parser.transform>
    gEconpy.parser.transform.expand_time_indices <gEconpy.parser.transform.expand_time_indices>
    gEconpy.parser.transform.to_block <gEconpy.parser.transform.to_block>
    gEconpy.parser.transform.to_distribution <gEconpy.parser.transform.to_distribution>
    gEconpy.parser.transform.to_sympy <gEconpy.parser.transform.to_sympy>
    gEconpy.plotting <gEconpy.plotting>
    gEconpy.pytensorf <gEconpy.pytensorf>
    gEconpy.pytensorf.block <gEconpy.pytensorf.block>
    gEconpy.pytensorf.compile <gEconpy.pytensorf.compile>
    gEconpy.pytensorf.real <gEconpy.pytensorf.real>
    gEconpy.pytensorf.real_eig <gEconpy.pytensorf.real_eig>
    gEconpy.solvers.backward_looking <gEconpy.solvers.backward_looking>
    gEconpy.solvers.cycle_reduction <gEconpy.solvers.cycle_reduction>
    gEconpy.solvers.gensys <gEconpy.solvers.gensys>
    gEconpy.solvers.shared <gEconpy.solvers.shared>
    gEconpy.solvers.sparse_root <gEconpy.solvers.sparse_root>
    gEconpy.solvers.sparse_root.base <gEconpy.solvers.sparse_root.base>
    gEconpy.solvers.sparse_root.direction <gEconpy.solvers.sparse_root.direction>
    gEconpy.solvers.sparse_root.dogleg <gEconpy.solvers.sparse_root.dogleg>
    gEconpy.solvers.sparse_root.gauss_newton <gEconpy.solvers.sparse_root.gauss_newton>
    gEconpy.solvers.sparse_root.globalization <gEconpy.solvers.sparse_root.globalization>
    gEconpy.solvers.sparse_root.levenberg_marquardt <gEconpy.solvers.sparse_root.levenberg_marquardt>
    gEconpy.solvers.sparse_root.line_search <gEconpy.solvers.sparse_root.line_search>
    gEconpy.solvers.sparse_root.sparse_root <gEconpy.solvers.sparse_root.sparse_root>
    gEconpy.utilities <gEconpy.utilities>
