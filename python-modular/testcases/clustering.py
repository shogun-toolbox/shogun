"""
Test Clustering
"""

from shogun.Distance import EuclidianDistance
from shogun.Clustering import *
from shogun.Library import Math_init_random

import util

def _clustering (indata):
	if indata.has_key('clustering_k'):
		first_arg=indata['clustering_k']
	elif indata.has_key('clustering_merges'):
		first_arg=indata['clustering_merges']
	else:
		return False

	fun=eval('util.get_feats_'+indata['feature_class'])
	feats=fun(indata)

	dargs=util.get_args(indata, 'distance_arg')
	dfun=eval(indata['distance_name'])
	distance=dfun(feats['train'], feats['train'], *dargs)

	fun=eval(indata['name'])
	clustering=fun(first_arg, distance)
	clustering.train()

	distance.init(feats['train'], feats['test'])

	if indata.has_key('clustering_radi'):
		radi=max(abs(clustering.get_radi()-indata['clustering_radi']))
		centers=max(abs(clustering.get_centers()- \
			indata['clustering_centers']).flat)
		return util.check_accuracy(indata['clustering_accuracy'],
			radi=radi, centers=centers)
	elif indata.has_key('clustering_merge_distance'):
		merge_distance=max(abs(clustering.get_merge_distance()- \
			indata['clustering_merge_distance']))
		pairs=max(abs(clustering.get_pairs()- \
			indata['clustering_pairs']).flat)
		return util.check_accuracy(indata['clustering_accuracy'],
			merge_distance=merge_distance, pairs=pairs)
	else:
		return util.check_accuracy(indata['clustering_accuracy'])

########################################################################
# public
########################################################################

def test (indata):
	Math_init_random(indata['clustering_init_random'])
	return _clustering(indata)

