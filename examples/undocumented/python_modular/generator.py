import os
import pickle
#tests = os.listdir("./Test")

tests=['kernel_linear_word_modular.py',
'kernel_gaussian_shift_modular.py',
'kernel_io_modular.py',
'kernel_auc_modular.py',
'kernel_gaussian_modular.py',
'kernel_histogram_word_string_modular.py',
'classifier_custom_kernel_modular.py',
'kernel_custom_modular.py',
'kernel_locality_improved_string_modular.py',
'kernel_local_alignment_string_modular.py',
'kernel_comm_ulong_string_modular.py',
'kernel_fixed_degree_string_modular.py',
'kernel_linear_modular.py',
'kernel_comm_word_string_modular.py',
'kernel_combined_custom_poly_modular.py',
'kernel_weighted_degree_position_string_modular.py',
'kernel_const_modular.py',
'classifier_knn_modular.py',
'kernel_distance_modular.py',
'kernel_linear_byte_modular.py',
'kernel_diag_modular.py',
'kernel_fisher_modular.py',
'kernel_chi2_modular.py',
'kernel_linear_string_modular.py',
'structure_dynprog_modular.py']




for i in tests:
    if i.endswith(".py"):
        mod_name = i[0:len(i)-3]
        mod = __import__(mod_name)
        f = open('./example_files/' + mod_name+  '.txt',"w")
#        print mod_name +'.txt'
#        print os.path.exists(mod_name +'.txt')
        for j in mod.parameter_list:
            a =  getattr(mod, mod_name)(*j)
            pickle.dump(a,f)

f = open('./example_files/' + mod_name+  '.txt')
pickle.load(f)





