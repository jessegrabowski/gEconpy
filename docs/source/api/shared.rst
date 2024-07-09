.. _api_shared:

******
Shared
******

--------------
Dynare Convert
--------------

.. automodule:: gEconpy.shared.dynare_convert

.. autosummary::
    :toctree: generated/

    get_name
    build_hash_table
    substitute_equation_from_dict
    make_var_to_matlab_sub_dict
    convert_var_timings_to_matlab
    write_lines_from_list
    make_mod_file


-------------------
Statsmodels Convert
-------------------

.. automodule:: gEconpy.shared.statsmodel_convert

.. autosummary::
    :toctree: generated/

    compile_to_statsmodels


---------
Utilities
---------

.. automodule:: gEconpy.shared.utilities

.. autosummary::
    :toctree: generated/

    flatten_list
    set_equality_equals_zero
    eq_to_ss
    expand_subs_for_all_times
    step_equation_forward
    step_equation_backward
    diff_through_time
    substitute_all_equations
    is_variable
    is_number
    sequential
    unpack_keys_and_values
    reduce_system_via_substitution
    merge_dictionaries
    make_all_var_time_combos
    build_Q_matrix
    compute_autocorrelation_matrix
    get_shock_std_priors_from_hyperpriors
    split_random_variables
