"""
Test Clustering
"""

from sg import sg
import util


def _set_clustering (indata):
	cname=util.fix_clustering_name_inconsistency(indata['clustering_name'])
	sg('new_clustering', cname)


def _train (indata, prefix):
	if indata.has_key(prefix+'max_iter'):
		max_iter=indata[prefix+'max_iter']
	else:
		max_iter=1000

	if indata.has_key(prefix+'k'):
		first_arg=indata[prefix+'k']
	elif indata.has_key(prefix+'merges'):
		first_arg=indata[prefix+'merges']
	else:
		raise StandardError, 'Incomplete clustering data.'

	sg('train_clustering', first_arg, max_iter)


def _evaluate (indata, prefix):
	if indata.has_key(prefix+'radi'):
		[radi, centers]=sg('get_clustering')
		radi=max(abs(radi.T[0]-indata[prefix+'radi']))
		centers=max(abs(centers-indata[prefix+'centers']).flat)

		return util.check_accuracy(indata[prefix+'accuracy'],
			radi=radi, centers=centers)

	elif indata.has_key(prefix+'merge_distance'):
		[merge_distances, pairs]=sg('get_clustering')
		merge_distances=max(abs(merge_distances.T[0]- \
			indata[prefix+'merge_distance']))
		pairs=max(abs(pairs-indata[prefix+'pairs']).flat)

		return util.check_accuracy(indata[prefix+'accuracy'],
			merge_distances=merge_distances, pairs=pairs)

	else:
		raise StandardError, 'Incomplete clustering data.'

########################################################################
# public
########################################################################

def test (indata):
	try:
		util.set_features(indata, 'distance_')
	except NotImplementedError, e:
		print e
		return True

	util.set_and_train_distance(indata)
	_set_clustering(indata)

	prefix='clustering_'
	try:
		_train(indata, prefix)
		return _evaluate(indata, prefix)
	except StandardError, e:
		print e
		return False

