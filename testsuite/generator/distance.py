from numpy import *
from shogun.Distance import *

import fileop
import dataop
import featop

def _compute (name, feats, data, *args):
	fun=eval(name)
	distance=fun(feats['train'], feats['train'], *args)
	dm_train=distance.get_distance_matrix()
	distance.init(feats['train'], feats['test'])
	dm_test=distance.get_distance_matrix()

	output={
		'name':name,
		'dm_train':dm_train,
		'dm_test':dm_test,
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test'])
	}
	output.update(fileop.get_output_params(name, fileop.T_DISTANCE, args))

	fileop.write(fileop.T_DISTANCE, output)

def _run_feats_real ():
	data=dataop.get_rand()
	feats=featop.get_simple('Real', data)

	_compute('EuclidianDistance', feats, data)
	_compute('CanberraMetric', feats, data)
	_compute('ChebyshewMetric', feats, data)
	_compute('GeodesicMetric', feats, data)
	_compute('JensenMetric', feats, data)
	_compute('ManhattanMetric', feats, data)
	_compute('MinkowskiMetric', feats, data, 1.3)

	feats=featop.get_simple('Real', data, sparse=True)
	_compute('SparseEuclidianDistance', feats, data)

def _run_feats_string_complex ():
	data=dataop.get_dna()
	feats=featop.get_string_complex('Word', data)

	_compute('CanberraWordDistance', feats, data)
	_compute('HammingWordDistance', feats, data, False)
	_compute('HammingWordDistance', feats, data, True)
	_compute('ManhattanWordDistance', feats, data)

def run ():
	_run_feats_real()
	_run_feats_string_complex()

