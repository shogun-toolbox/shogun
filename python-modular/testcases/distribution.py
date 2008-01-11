"""
Test Distribution
"""
from numpy import inf, nan
from shogun.Distribution import *

import util

def _distribution (indata):
	fun=eval('util.get_feats_'+indata['feature_class'])
	feats=fun(indata)

	if indata['name']=='HMM':
		distribution=HMM(feats['train'], indata['distribution_N'],
			indata['distribution_M'], indata['distribution_pseudo'])
		distribution.train()
		distribution.baum_welch_viterbi_train(BW_NORMAL)
	else:
		fun=eval(indata['name'])
		distribution=fun(feats['train'])
		distribution.train()

	likelihood=distribution.get_log_likelihood_sample()
	num_examples=feats['train'].get_num_vectors()
	num_param=distribution.get_num_model_parameters()
	derivatives=0
	for i in xrange(num_param):
		for j in xrange(num_examples):
			val=distribution.get_log_derivative(i, j)
			if val!=-inf and val!=nan: # only consider sparse matrix!
				derivatives+=val

	derivatives=abs(derivatives-indata['distribution_derivatives'])
	likelihood=abs(likelihood-indata['distribution_likelihood'])

	if indata['name']=='HMM':
		best_path=0
		best_path_state=0
		for i in xrange(indata['distribution_examples']):
			best_path+=distribution.best_path(i)
			for j in xrange(indata['distribution_N']):
				best_path_state+=distribution.get_best_path_state(i, j)

		best_path=abs(best_path-indata['distribution_best_path'])
		best_path_state=abs(best_path_state-\
			indata['distribution_best_path_state'])

		return util.check_accuracy(indata['distribution_accuracy'],
			derivatives=derivatives, likelihood=likelihood,
			best_path=best_path, best_path_state=best_path_state)
	else:
		return util.check_accuracy(indata['distribution_accuracy'],
			derivatives=derivatives, likelihood=likelihood)

########################################################################
# public
########################################################################

def test (indata):
	return _distribution(indata)

