"""
Test Clustering
"""

from sg import sg
import util


def _evaluate (indata):
	if indata.has_key('clustering_radi'):
		[radi, centers]=sg('get_clustering')
		radi=max(abs(radi.T[0]-indata['clustering_radi']))
		centers=max(abs(centers-indata['clustering_centers']).flat)

		return util.check_accuracy(indata['clustering_accuracy'],
			radi=radi, centers=centers)

	elif indata.has_key('clustering_merge_distance'):
		[merge_distances, pairs]=sg('get_clustering')
		print merge_distances.T[0]
		print indata['clustering_merge_distance']
		merge_distances=max(abs(merge_distances.T[0]- \
			indata['clustering_merge_distance']))
		pairs=max(abs(pairs-indata['clustering_pairs']).flat)

		return util.check_accuracy(indata['clustering_accuracy'],
			merge_distances=merge_distances, pairs=pairs)

	else:
		return util.check_accuracy(indata['clustering_accuracy'])

########################################################################
# public
########################################################################

def test (indata):
	util.set_features(indata)
	util.set_and_train_distance(indata)

	cname=util.fix_clustering_name_inconsistency(indata['name'])
	sg('new_clustering', cname)

	if indata.has_key('clustering_k'):
		if indata.has_key('clustering_max_iter'):
			max_iter=indata['clustering_max_iter']
		else:
			max_iter=1000
		sg('train_clustering', indata['clustering_k'], max_iter)

	elif indata.has_key('clustering_merges'):
		sg('train_clustering', indata['clustering_merges'])

	else:
		print 'Incomplete clustering data.'
		return False

	return _evaluate(indata)

