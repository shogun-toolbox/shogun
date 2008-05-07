"""Generator for Clustering
"""

from numpy import matrix
from shogun.Distance import EuclidianDistance
import shogun.Clustering as clustering

import fileop
import featop
import dataop
from config import CLUSTERING, C_CLUSTERING, C_DISTANCE

def _get_outdata (name, params):
	"""Return data to be written into the testcase's file.

	After computations and such, the gathered data is structured and
	put into one data structure which can conveniently be written to a
	file that will represent the testcase.
	
	@param name Clustering method's name
	@param params Gathered data
	@return Dict containing testcase data to be written to file
	"""

	outdata={
		'name':name,
		'data_train':matrix(params['data']['train']),
		'data_test':matrix(params['data']['test']),
		'clustering_accuracy':CLUSTERING[name][0],
		'init_random':dataop.INIT_RANDOM,
	}

	optional=['k', 'radi', 'centers', 'merges', 'merge_distance', 'pairs']
	for opt in optional:
		if params.has_key(opt):
			outdata['clustering_'+opt]=params[opt]

	outdata['distance_name']=params['dname']
	dparams=fileop.get_outdata(params['dname'], C_DISTANCE, params['dargs'])
	outdata.update(dparams)

	return outdata

def _run (name, first_arg):
	"""Run generator for a specific clustering method.

	@param name Name of the clustering method to run.
	@param first_arg First argument to the clustering's constructor; so far, only this distinguishes the instantion of the different methods.
	"""

	params={
		first_arg:3,
		'dname':'EuclidianDistance',
		'dargs':[],
	}
	params['data']=dataop.get_clouds(params[first_arg], 5)
	feats=featop.get_simple('Real', params['data'])
	dfun=eval(params['dname'])
	distance=dfun(feats['train'], feats['train'], *params['dargs'])

	fun=eval('clustering.'+name)
	clustering=fun(params[first_arg], distance)
	clustering.train()

	distance.init(feats['train'], feats['test'])

	if name=='KMeans':
		params['radi']=clustering.get_radiuses()
		params['centers']=clustering.get_cluster_centers()
	elif name=='Hierarchical':
		params['merge_distance']=clustering.get_merge_distances()
		params['pairs']=clustering.get_cluster_pairs()

	outdata=_get_outdata(name, params)
	fileop.write(C_CLUSTERING, outdata)


def run ():
	"""Run all clustering methods."""

	_run('KMeans', 'k')
	_run('Hierarchical', 'merges')
