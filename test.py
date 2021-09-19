from gEcon import *
from model_utilities import *

gcn_filepath = 'GCN Files/Test_Model.gcn'

model, param_dict, shocks, var_list, try_reduce_vars = parse_gcn_file(gcn_filepath)

ss_model, ss_var_list = convert_model_to_steady_state(model, var_list)
ss_model = rewrite_model_equations_eq_0(ss_model)
ss_model = substitute_all_params(ss_model, param_dict)

f_model, f_jac = convert_model_to_numpy_function(ss_model, ss_var_list, return_jac=True)

result, ss_value_dict = compute_ss(f_model, ss_var_list)

for key, value in sorted(ss_value_dict.items(), key=lambda x: x[1], reverse=True):
    print(f'{str(key):<15} {value:0.5}')
