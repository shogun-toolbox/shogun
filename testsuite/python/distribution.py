"""
Test Distribution
"""
from numpy import inf, nan
from sg import sg
import util


########################################################################
# public
########################################################################

def test (indata):
	util.set_features(indata)

	if indata['name']=='HMM':
		sg('new_hmm', indata['distribution_N'], indata['distribution_M'])
		sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', indata['order'])
		sg('bw')
	else:
		print 'Can\'t yet train other distributions in static interface.'
		return False

	# what is sg('likelihood')?
	likelihood=abs(sg('hmm_likelihood')-indata['distribution_likelihood'])
	return util.check_accuracy(indata['distribution_accuracy'],
		likelihood=likelihood)

	# best path? which? no_b_trans? trans? trans_deriv?
	if indata['name']=='HMM':
		best_path=0
		best_path_state=0
		for i in xrange(indata['distribution_num_examples']):
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

