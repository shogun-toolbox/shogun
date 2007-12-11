from numpy import *
from shogun.Clustering import *
from shogun.Distance import EuclidianDistance

import fileop
import featop
import dataop
from config import CLUSTERING, T_CLUSTERING, T_DISTANCE

def _get_output_params (name, params, data):
	output={
		'name':name,
		'data_train':matrix(data['data']['train']),
		'data_test':matrix(data['data']['test']),
		'accuracy':CLUSTERING[name][0],
	}

	for k, v in params.iteritems():
		output['clustering_'+k]=v

	output['distance_name']=data['dname']
	dparams=fileop.get_output_params(
		data['dname'], T_DISTANCE, data['dargs'])
	output.update(dparams)

	return output

def _run (name, first_arg):
	params={
		first_arg:3,
	}
	data={
		'dname':'EuclidianDistance',
		'dargs':[],
		'data':dataop.get_rand(),
	}
	feats=featop.get_simple('Real', data['data'])
	dfun=eval(data['dname'])
	distance=dfun(feats['train'], feats['train'], *data['dargs'])

	fun=eval(name)
	clustering=fun(params[first_arg], distance)
	clustering.train()

	distance.init(feats['train'], feats['test'])
	#params['classified']=clustering.classify().get_labels()

	output=_get_output_params(name, params, data)
	fileop.write(T_CLUSTERING, output)


def run ():
	_run('KMeans', 'k')
	_run('Hierarchical', 'merges')
