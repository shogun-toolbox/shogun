"""
Test Clustering
"""

from sg import sg
import util


def _set_clustering (indata):
	cname=util.fix_clustering_name_inconsistency(indata['name'])
	sg('new_clustering', cname)


def _train (indata):
	if indata.has_key('clustering_max_iter'):
		max_iter=indata['clustering_max_iter']
	else:
		max_iter=1000

	if indata.has_key('clustering_k'):
		first_arg=indata['clustering_k']
	elif indata.has_key('clustering_merges'):
		first_arg=indata['clustering_merges']
	else:
		raise StandardError, 'Incomplete clustering data.'

	sg('train_clustering', first_arg, max_iter)


def _evaluate (indata):
	if indata.has_key('clustering_radi'):
		[radi, centers]=sg('get_clustering')
		radi=max(abs(radi.T[0]-indata['clustering_radi']))
		centers=max(abs(centers-indata['clustering_centers']).flat)

		return util.check_accuracy(indata['clustering_accuracy'],
			radi=radi, centers=centers)

	elif indata.has_key('clustering_merge_distance'):
		[merge_distances, pairs]=sg('get_clustering')
		merge_distances=max(abs(merge_distances.T[0]- \
			indata['clustering_merge_distance']))
		pairs=max(abs(pairs-indata['clustering_pairs']).flat)

		return util.check_accuracy(indata['clustering_accuracy'],
			merge_distances=merge_distances, pairs=pairs)

	else:
		raise StandardError, 'Incomplete clustering data.'

########################################################################
# public
########################################################################

def test (indata):
	try:
		util.set_features(indata)
	except NotImplementedError, e:
		print e
		return True

	util.set_and_train_distance(indata)
	_set_clustering(indata)

	try:
		_train(indata)
		return _evaluate(indata)
	except StandardError, e:
		print e
		return False

