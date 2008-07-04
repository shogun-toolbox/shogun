"""
Test Regression
"""

from sg import sg
from numpy import double
import util


def _set_regression (indata):
	sg('threads', indata['regression_num_threads'])
	sg('set_labels', 'TRAIN', double(indata['regression_labels']))

	rname=util.fix_regression_name_inconsistency(indata['name'])
	sg('new_regression', rname)


def _train (indata):
	if indata['regression_type']=='svm':
		sg('c', double(indata['regression_C']))
		sg('svm_epsilon', indata['regression_epsilon'])
		sg('svr_tube_epsilon', indata['regression_tube_epsilon'])
	elif indata['regression_type']=='kernelmachine':
		sg('krr_tau', indata['regression_tau'])
	else:
		raise StandardError, 'Incomplete regression data.'

	sg('train_regression')


def _evaluate (indata):
	alphas=0
	bias=0
	sv=0

	if indata.has_key('regression_bias'):
		[bias, weights]=sg('get_svm')
		weights=weights.T
		bias=abs(bias-indata['regression_bias'])
		alphas=max(abs(weights[0]-indata['regression_alphas']))
		sv=max(abs(weights[1]-indata['regression_support_vectors']))

	sg('init_kernel', 'TEST')
	classified=max(abs(sg('classify')-indata['regression_classified']))

	return util.check_accuracy(indata['regression_accuracy'],
		alphas=alphas, bias=bias, sv=sv, classified=classified)


########################################################################
# public
########################################################################

def test (indata):
	try:
		util.set_features(indata)
	except NotImplementedError, e:
		print e
		return True

	util.set_and_train_kernel(indata)

	try:
		_set_regression(indata)
	except RuntimeError, e:
		print "%s is disabled/unavailable!" % indata['name']
		return True

	try:
		_train(indata)
	except StandardError, e:
		print e
		return False

	return _evaluate(indata)

