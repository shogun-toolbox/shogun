"""
Test Clustering
"""

from modshogun import EuclideanDistance
from modshogun import *

import util


def _evaluate (indata):
	if 'clustering_k' in indata:
		first_arg=indata['clustering_k']
	elif 'clustering_merges' in indata:
		first_arg=indata['clustering_merges']
	else:
		return False

	feats=util.get_features(indata, 'distance_')
	dfun=eval(indata['distance_name'])
	distance=dfun(feats['train'], feats['train'])

	cfun=eval(indata['clustering_name'])
	clustering=cfun(first_arg, distance)
	clustering.train()

	if 'clustering_radi' in indata:
		radi=max(abs(clustering.get_radiuses()-indata['clustering_radi']))
		centers=max(abs(clustering.get_cluster_centers().flatten() - \
			indata['clustering_centers'].flat))
		return util.check_accuracy(indata['clustering_accuracy'],
			radi=radi, centers=centers)
	elif 'clustering_merge_distance' in indata:
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

