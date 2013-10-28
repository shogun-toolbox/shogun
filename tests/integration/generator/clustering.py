"""Generator for Clustering
"""

from numpy import matrix
from shogun.Distance import EuclideanDistance
import shogun.Clustering as clustering

from shogun.Library import Math_init_random

import fileop
import featop
import dataop
import category


def _run (name, first_arg):
	"""
	Run generator for a specific clustering method.

	@param name Name of the clustering method to run.
	@param first_arg First argument to the clustering's constructor; so far, only this distinguishes the instantion of the different methods.
	"""

	# put some constantness into randomness
	Math_init_random(dataop.INIT_RANDOM)

	num_clouds=3
	params={
		'name': 'EuclideanDistance',
		'data': dataop.get_clouds(num_clouds, 5),
		'feature_class': 'simple',
		'feature_type': 'Real'
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])
	dfun=eval(params['name'])
	distance=dfun(feats['train'], feats['train'])
	output=fileop.get_output(category.DISTANCE, params)

	params={
		'name': name,
		'accuracy': 1e-8,
		first_arg: num_clouds
	}
	fun=eval('clustering.'+name)
	clustering=fun(params[first_arg], distance)
	clustering.train()

	if name=='KMeans':
		params['radi']=clustering.get_radiuses()
		params['centers']=clustering.get_cluster_centers()
	elif name=='Hierarchical':
		params['merge_distance']=clustering.get_merge_distances()
		params['pairs']=clustering.get_cluster_pairs()

	output.update(fileop.get_output(category.CLUSTERING, params))
	fileop.write(category.CLUSTERING, output)


def run ():
	"""Run all clustering methods."""

	_run('KMeans', 'k')
	_run('Hierarchical', 'merges')
