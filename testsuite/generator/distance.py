from numpy import *
from shogun.Distance import *

import fileop
import dataop
import featop

def _compute (name, feats, data, *args):
	dfun=eval(name)
	d=dfun(feats['train'], feats['train'], *args)
	dm_train=d.get_distance_matrix()
	d.init(feats['train'], feats['test'])
	dm_test=d.get_distance_matrix()

 	output={
		'dm_train':dm_train,
		'dm_test':dm_test,
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test'])
	}
	output.update(fileop.get_output_params(name, args))

	return [name, output]

def _run_feats_real ():
	data=dataop.get_rand()
	feats=featop.get_simple('Real', data)

	fileop.write(_compute('EuclidianDistance', feats, data))
	fileop.write(_compute('CanberraMetric', feats, data))
	fileop.write(_compute('ChebyshewMetric', feats, data))
	fileop.write(_compute('GeodesicMetric', feats, data))
	fileop.write(_compute('JensenMetric', feats, data))
	fileop.write(_compute('ManhattanMetric', feats, data))
	fileop.write(_compute('MinkowskiMetric', feats, data, 1.3))

	feats=featop.get_simple('Real', data, sparse=True)
	fileop.write(_compute('SparseEuclidianDistance', feats, data))

def _run_feats_string_complex ():
	data=dataop.get_dna()
	feats=featop.get_string_complex('Word', data)

	fileop.write(_compute('CanberraWordDistance', feats, data))
	fileop.write(_compute('HammingWordDistance', feats, data, False))
	fileop.write(_compute('HammingWordDistance', feats, data, True))
	fileop.write(_compute('ManhattanWordDistance', feats, data))

def run ():
	fileop.TYPE='Distance'

	_run_feats_real()
	_run_feats_string_complex()

