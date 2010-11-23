import os
import pickle
#tests = os.listdir("./Test")

test_dir='../../../testsuite/tests'
tests=['kernel_gaussian_modular.py',
        'kernel_fixed_degree_string_modular.py']

#tests=['kernel_linear_word_modular.py',
#'kernel_gaussian_shift_modular.py',
#'kernel_io_modular.py',
#'kernel_auc_modular.py',
#'kernel_gaussian_modular.py',
#'kernel_histogram_word_string_modular.py',
#'classifier_custom_kernel_modular.py',
#'kernel_custom_modular.py',
#'kernel_locality_improved_string_modular.py',
#'kernel_local_alignment_string_modular.py',
#'kernel_comm_ulong_string_modular.py',
#'kernel_fixed_degree_string_modular.py',
#'kernel_linear_modular.py',
#'kernel_comm_word_string_modular.py',
#'kernel_combined_custom_poly_modular.py',
#'kernel_weighted_degree_position_string_modular.py',
#'kernel_const_modular.py',
#'classifier_knn_modular.py',
#'kernel_distance_modular.py',
#'kernel_linear_byte_modular.py',
#'kernel_diag_modular.py',
#'kernel_fisher_modular.py',
#'kernel_chi2_modular.py',
#'kernel_linear_string_modular.py',
#'structure_dynprog_modular.py']


def get_fname(mod_name, i):
    return os.path.join(test_dir, mod_name + str(i) + '.txt')

def generator():
    for t in tests:
        if t.endswith(".py"):
            mod_name = t[:-3]
            print mod_name
            mod = __import__(mod_name)
            for i in xrange(len(mod.parameter_list)):
                fname = get_fname(mod_name, i)
                f = open(fname, "w")
                par=mod.parameter_list[i]
                a =  getattr(mod, mod_name)(*par)
                pickle.dump(a,f)

if __name__=='__main__':
    generator()
