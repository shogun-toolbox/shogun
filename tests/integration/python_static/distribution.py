"""
Test Distribution
"""
from numpy import inf, nan
from sg import sg
import util

def _evaluate (indata):
	prefix='distribution_'
	# what is sg('likelihood')?
	likelihood=abs(sg('hmm_likelihood')-indata[prefix+'likelihood'])
	return util.check_accuracy(indata[prefix+'accuracy'],
		likelihood=likelihood)

	# best path? which? no_b_trans? trans? trans_deriv?
	if indata['name']=='HMM':
		best_path=0
		best_path_state=0
		for i in xrange(indata[prefix+'num_examples']):
			best_path+=distribution.best_path(i)
			for j in xrange(indata[prefix+'N']):
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


def _set_distribution (indata):
	prefix='distribution_'
	if indata[prefix+'name']=='HMM':
		sg('new_hmm', indata[prefix+'N'], indata[prefix+'M'])
		sg('bw')
	else:
		raise NotImplementedError, 'Can\'t yet train other distributions than HMM in static interface.'


########################################################################
# public
########################################################################

def test (indata):
	try:
		util.set_features(indata, 'distribution_')
		_set_distribution(indata)
	except NotImplementedError, e:
		print e
		return True

	return _evaluate(indata)

