"""
Test Regression
"""

from numpy import double
from shogun.Features import Labels
from shogun.Kernel import *
from shogun.Regression import *

import util

def _regression (indata):
	fun=eval('util.get_feats_'+indata['feature_class'])
	feats=fun(indata)

	kargs=util.get_args(indata, 'kernel_arg')
	fun=eval(indata['kernel_name']+'Kernel')
	kernel=fun(feats['train'], feats['train'], *kargs)
	kernel.parallel.set_num_threads(indata['regression_num_threads'])

	try:
		fun=eval(indata['name'])
	except NameError, e:
		print "%s is disabled/unavailable!"%indata['name']
		return False

	labels=Labels(double(indata['regression_labels']))
	if indata['regression_type']=='svm':
		regression=fun(indata['regression_C'], indata['regression_epsilon'],
			kernel, labels)
	elif indata['regression_type']=='kernelmachine':
		regression=fun(indata['regression_tau'], kernel, labels)
	else:
		return False

	regression.parallel.set_num_threads(indata['regression_num_threads'])
	if indata.has_key('regression_tube_epsilon'):
		regression.set_tube_epsilon(indata['regression_tube_epsilon'])

	regression.train()

	alphas=0
	bias=0
	sv=0
	if indata.has_key('regression_bias'):
		bias=abs(regression.get_bias()-indata['regression_bias'])
	if indata.has_key('regression_alphas'):
		for item in regression.get_alphas().tolist():
			alphas+=item
		alphas=abs(alphas-indata['regression_alphas'])
	if indata.has_key('regression_support_vectors'):
		for item in inregression.get_support_vectors().tolist():
			sv+=item
		sv=abs(sv-indata['regression_support_vectors'])

	kernel.init(feats['train'], feats['test'])
	classified=max(abs(
		regression.classify().get_labels()-indata['regression_classified']))

	return util.check_accuracy(indata['regression_accuracy'], alphas=alphas,
		bias=bias, support_vectors=sv, classified=classified)

########################################################################
# public
########################################################################

def test (indata):
	return _regression(indata)

