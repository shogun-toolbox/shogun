"""
Test Distribution
"""

from shogun.Distribution import *

import util

def _distribution (indata):
	fun=eval('util.get_feats_'+indata['feature_class'])
	feats=fun(indata)

	if indata['name']=='HMM':
		model=Model()
		distribution=HMM(indata['distribution_N'], indata['distribution_M'],
			model, indata['distribution_pseudo'])
	else:
		fun=eval(indata['name'])
		distribution=fun(feats['train'])

	distribution.train()

	if indata['name']=='Histogram':
		num_examples=feats['train'].get_num_vectors()
		num_param=distribution.get_num_model_parameters()
		derivatives=0
		likelihood=0
		for i in xrange(num_examples):
			for j in xrange(num_param):
				derivatives+=distribution.get_log_derivative(j, i)
			likelihood+=distribution.get_log_likelihood_example(i)

		derivatives=abs(derivatives-indata['distribution_derivatives'])
		likelihood=abs(likelihood-indata['distribution_likelihood'])

		return util.check_accuracy(indata['distribution_accuracy'],
			derivatives=derivatives, likelihood=likelihood)
	else:
		return util.check_accuracy(indata['distribution_accuracy'])

########################################################################
# public
########################################################################

def test (indata):
	return _distribution(indata)

