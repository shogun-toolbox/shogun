from numpy import double
from shogun.Features import Labels
from shogun.Kernel import *
from shogun.Regression import *

import util

def _regression (input):
	fun=eval('util.get_feats_'+input['feature_class'])
	feats=fun(input)

	kargs=util.get_args(input, 'kernel_arg')
	fun=eval(input['kernel_name']+'Kernel')
	kernel=fun(feats['train'], feats['train'], *kargs)
	kernel.parallel.set_num_threads(input['regression_num_threads'])

	fun=eval(input['name'])
	labels=Labels(double(input['regression_labels']))
	if input['regression_type']=='svm':
			regression=fun(input['regression_C'], input['regression_epsilon'],
				kernel, labels)
	elif input['regression_type']=='kernelmachine':
			regression=fun(input['regression_tau'], kernel, labels)
	else:
		return False

	regression.parallel.set_num_threads(input['regression_num_threads'])
	if input.has_key('regression_tube_epsilon'):
		regression.set_tube_epsilon(input['regression_tube_epsilon'])

	regression.train()

	alphas=0
	bias=0
	sv=0
	if input['regression_num_threads']==1:
		if input.has_key('regression_bias'):
			bias=abs(regression.get_bias()-input['regression_bias'])
		if input.has_key('regression_alphas'):
			alphas=max(abs(regression.get_alphas()-input['regression_alphas']))
		if input.has_key('regression_support_vectors'):
			sv=max(abs(regression.get_support_vectors()-input['regression_support_vectors']))
	else: # lower accuracy
		accuracy=1e-4

	kernel.init(feats['train'], feats['test'])
	classified=max(abs(
		regression.classify().get_labels()-input['regression_classified']))

	return util.check_accuracy(input['regression_accuracy'],
		alphas=alphas, bias=bias, sv=sv, classified=classified)

########################################################################
# public
########################################################################

def test (input):
	return _regression(input)

