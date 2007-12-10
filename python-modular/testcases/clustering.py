from shogun.Distance import EuclidianDistance
from shogun.Clustering import *

import util

def _clustering (input):
	if input.has_key('clustering_k'):
		first_arg=input['clustering_k']
	elif input.has_key('clustering_merges'):
		first_arg=input['clustering_merges']
	else:
		return False

	fun=eval('util.get_feats_'+input['feature_class'])
	feats=fun(input)

	dargs=util.get_args(input, 'distance_arg')
	dfun=eval(input['distance_name'])
	distance=dfun(feats['train'], feats['train'], *dargs)
	distance.parallel.set_num_threads(input['clustering_num_threads'])

	fun=eval(input['name'])
	clustering=fun(first_arg, distance)
	clustering.parallel.set_num_threads(input['clustering_num_threads'])
	clustering.train()

	distance.init(feats['train'], feats['test'])
	#classified=max(abs(clustering.classify().get_labels()-input['clustering_classified']))
	classified=0

	return util.check_accuracy(
		input['clustering_accuracy'], classified=classified)

########################################################################
# public
########################################################################

def test (input):
	return _clustering(input)

