"""
Test Regression
"""

from sg import sg
from numpy import double
import util

########################################################################
# public
########################################################################

def test (indata):
	util.set_features(indata)
	util.set_and_train_kernel(indata)
	sg('threads', indata['regression_num_threads'])
	sg('set_labels', 'TRAIN', double(indata['regression_labels']))

	try:
		sg('new_regression',
			util.fix_regression_name_inconsistency(indata['name']))
	except RuntimeError, e:
		print "%s is disabled/unavailable!" % indata['name']
		return True

	if indata['regression_type']=='svm':
		sg('c', double(indata['regression_C']))
		sg('svm_epsilon', indata['regression_epsilon'])
		sg('svr_tube_epsilon', indata['regression_tube_epsilon'])
	elif indata['regression_type']=='kernelmachine':
		sg('krr_tau', indata['regression_tau'])
	else:
		return False

	sg('train_regression')

	alphas=0
	bias=0
	support_vectors=0
	if indata.has_key('regression_bias'):
		[bias, weights]=sg('get_svm')
		weights=weights.T
		bias=abs(bias-indata['regression_bias'])
		alphas=max(abs(weights[0]-indata['regression_alphas']))
		support_vectors=max(abs(weights[1]-indata['regression_support_vectors']))


	sg('init_kernel', 'TEST')
	classified=max(abs(sg('classify')-indata['regression_classified']))

	return util.check_accuracy(indata['regression_accuracy'], alphas=alphas,
		bias=bias, support_vectors=support_vectors, classified=classified)

