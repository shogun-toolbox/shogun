"""
Generator for Clustering
"""

from numpy import *
from shogun.Clustering import *
from shogun.Distance import EuclidianDistance

import fileop
import featop
import dataop
from config import CLUSTERING, C_CLUSTERING, C_DISTANCE

def _get_outdata_params (name, params, data):
	outdata={
		'name':name,
		'data_train':matrix(data['data']['train']),
		'data_test':matrix(data['data']['test']),
		'accuracy':CLUSTERING[name][0],
	}

	for key, val in params.iteritems():
		outdata['clustering_'+key]=val

	outdata['distance_name']=data['dname']
	dparams=fileop.get_outdata_params(
		data['dname'], C_DISTANCE, data['dargs'])
	outdata.update(dparams)

	return outdata

def _run (name, first_arg):
	params={
		first_arg:3,
	}
	data={
		'dname':'EuclidianDistance',
		'dargs':[],
		'data':dataop.get_clouds(params[first_arg], 5),
	}
	feats=featop.get_simple('Real', data['data'])
	dfun=eval(data['dname'])
	distance=dfun(feats['train'], feats['train'], *data['dargs'])

	fun=eval(name)
	clustering=fun(params[first_arg], distance)
	clustering.train()

	distance.init(feats['train'], feats['test'])

	if name=='KMeans':
		params['radi']=clustering.get_radi()
		params['centers']=clustering.get_centers()
	elif name=='Hierarchical':
		params['merge_distance']=clustering.get_merge_distance()
		params['pairs']=clustering.get_pairs()

	outdata=_get_outdata_params(name, params, data)
	fileop.write(C_CLUSTERING, outdata)


def run ():
	_run('KMeans', 'k')
	_run('Hierarchical', 'merges')
