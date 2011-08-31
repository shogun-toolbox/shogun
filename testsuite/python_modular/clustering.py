"""
Test Clustering
"""

from shogun.Distance import EuclidianDistance
from shogun.Clustering import *

import util


def _evaluate (indata):
	if indata.has_key('clustering_k'):
		first_arg=indata['clustering_k']
	elif indata.has_key('clustering_merges'):
		first_arg=indata['clustering_merges']
	else:
		return False

	feats=util.get_features(indata, 'distance_')
	print feats['train'].get_num_vectors()
	dfun=eval(indata['distance_name'])
	distance=dfun(feats['train'], feats['train'])

	cfun=eval(indata['clustering_name'])
	clustering=cfun(first_arg, distance)
	clustering.train()

	print clustering.get_k()

	if indata.has_key('clustering_radi'):
		radi=max(abs(clustering.get_radiuses()-indata['clustering_radi']))
		centers=max(abs(clustering.get_cluster_centers().flatten() - \
			indata['clustering_centers'].flat))
		return util.check_accuracy(indata['clustering_accuracy'],
			radi=radi, centers=centers)
	elif indata.has_key('clustering_merge_distance'):
		merge_distance=max(abs(clustering.get_merge_distances()- \
			indata['clustering_merge_distance']))
		pairs=max(abs(clustering.get_cluster_pairs()- \
			indata['clustering_pairs']).flat)
		return util.check_accuracy(indata['clustering_accuracy'],
			merge_distance=merge_distance, pairs=pairs)
	else:
		return util.check_accuracy(indata['clustering_accuracy'])


########################################################################
# public
########################################################################

def test (indata):
	return _evaluate(indata)

