#!/usr/bin/env python

import os
import sys
import pickle

example_dir = '../../../examples/undocumented/python_modular'
test_dir = '../../../tests/integration/data/python%d-tests' % sys.version_info[0]
blacklist = ("__init__.py",
		"classifier_libsvm_minimal_modular.py",
		"classifier_multiclassocas_modular.py",
		"modelselection_grid_search_kernel.py",
		"serialization_string_kernels_modular.py",
		"kernel_fisher_modular.py",
		"kernel_top_modular.py",
		"distribution_hmm_modular.py",
		"kernel_director_linear_modular.py",
		"features_director_dot_modular.py",
		"converter_tdistributedstochasticneighborembedding_modular.py",
		"evaluation_director_contingencytableevaluation_modular.py",
		"distance_director_euclidean_modular.py",
		"classifier_multiclass_ecoc_random.py",
		"statistics_hsic.py",
		"transfer_multitask_clustered_logistic_regression.py",
		"mathematics_logdet.py",
		"classifier_svmlight_batch_linadd_modular.py",
#the tests below all fail on travis but work fine on our buildbot
#		"classifier_lda_modular.py",
#		"classifier_liblinear_modular.py",
#		"converter_tdistributedstochasticneighborembedding_modular.py",
#		"distance_mahalanobis_modular.py",
#		"mathematics_sparseinversecovariance_modular.py",
#		"preprocessor_dimensionreductionpreprocessor_modular.py",
#		"preprocessor_kernelpca_modular.py",
#		"preprocessor_pca_modular.py",
#		"regression_kernel_ridge_modular.py",
#		"regression_least_squares_modular.py",
#		"regression_linear_ridge_modular.py",
#		"regression_svrlight_modular.py",
#		"features_dense_protocols_modular.py",
#		"features_dense_zero_copy_modular.py",
#		"transfer_multitask_l12_logistic_regression.py",
#		"transfer_multitask_trace_logistic_regression.py",
		)

def get_fname(mod_name, i):
	return os.path.join(test_dir, mod_name + str(i) + '.txt')

def setup_tests(tests=[]):
	os.chdir(example_dir)

	if not len(tests):
		tests =  os.listdir(".")
		tests.sort()
	else:
		tests = [ os.path.basename(t) for t in tests ]

	sys.path.insert(0, '.')
	return tests

def check_for_function(fname):
	for l in open(fname).readlines():
		if l.startswith("def "):
			return True
	return False

def get_test_mod(t):
	if t.endswith(".py") and not t.startswith('.') and t not in blacklist:
		mod_name = t[:-3]

		if not check_for_function(t):
			raise Exception("ERROR (no function)")

		return __import__(mod_name), mod_name

def run_test(mod, mod_name, i):
	fname = get_fname(mod_name, i)
	par=mod.parameter_list[i]
	a =  getattr(mod, mod_name)(*par)
	return a

def generator(tests):
	for t in tests:
		try:
			mod, mod_name = get_test_mod(t)
		except TypeError:
			continue
		except Exception as e:
			print(t, e)
			continue
		fname = ""

		print("%-60s" % mod_name)
		#print("%+60s" % "...")
		try:
			for i in range(len(mod.parameter_list)):
				fname = get_fname(mod_name, i)
				a = run_test(mod, mod_name, i)
				pickle.dump(a,open(fname, "wb"),0)
			print("OK")
		except Exception as e:
			print("ERROR generating '%s' using '%s'" % (fname,t))
			print(e)
			continue

if __name__=='__main__':
	from optparse import OptionParser
	op=OptionParser()
	op.set_usage("[<file1> <file2> ...]")
	(opts, args)=op.parse_args()
	tests = setup_tests(args)
	generator(tests)
