"""
Test Distribution
"""
from numpy import inf, nan
from shogun.Distribution import *

import util

def _evaluate (indata):
	prefix='distribution_'
	feats=util.get_features(indata, prefix)

	if indata[prefix+'name']=='HMM':
		distribution=HMM(feats['train'], indata[prefix+'N'],
			indata[prefix+'M'], indata[prefix+'pseudo'])
		distribution.train()
		distribution.baum_welch_viterbi_train(BW_NORMAL)
	else:
		dfun=eval(indata[prefix+'name'])
		distribution=dfun(feats['train'])
		distribution.train()

	likelihood=distribution.get_log_likelihood_sample()
	num_examples=feats['train'].get_num_vectors()
	num_param=distribution.get_num_model_parameters()
	derivatives=0
	for i in range(num_param):
		for j in range(num_examples):
			val=distribution.get_log_derivative(i, j)
			if val!=-inf and val!=nan: # only consider sparse matrix!
				derivatives+=val

	derivatives=abs(derivatives-indata[prefix+'derivatives'])
	likelihood=abs(likelihood-indata[prefix+'likelihood'])

	if indata[prefix+'name']=='HMM':
		best_path=0
		best_path_state=0
		for i in range(indata[prefix+'num_examples']):
			best_path+=distribution.best_path(i)
			for j in range(indata[prefix+'N']):
				best_path_state+=distribution.get_best_path_state(i, j)

		best_path=abs(best_path-indata[prefix+'best_path'])
		best_path_state=abs(best_path_state-\
			indata[prefix+'best_path_state'])

		return util.check_accuracy(indata[prefix+'accuracy'],
			derivatives=derivatives, likelihood=likelihood,
			best_path=best_path, best_path_state=best_path_state)
	else:
		return util.check_accuracy(indata[prefix+'accuracy'],
			derivatives=derivatives, likelihood=likelihood)


########################################################################
# public
########################################################################

def test (indata):
	return _evaluate(indata)

