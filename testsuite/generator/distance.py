"""Generator for Distance"""

import numpy
import shogun.Distance as distance

import fileop
import dataop
import featop
import config

def _compute (name, feats, data, *args):
	"""Compute a distance and gather result data.

	@param name Name of the distance
	@param feats Train and test features
	@param data Train and test data (for output)
	@param *args variable argument list for distance's constructor
	"""

	fun=eval('distance.'+name)
	dist=fun(feats['train'], feats['train'], *args)
	dm_train=dist.get_distance_matrix()
	dist.init(feats['train'], feats['test'])
	dm_test=dist.get_distance_matrix()

	outdata={
		'name':name,
		'dm_train':dm_train,
		'dm_test':dm_test,
		'data_train':numpy.matrix(data['train']),
		'data_test':numpy.matrix(data['test'])
	}
	outdata.update(fileop.get_outdata(name, config.C_DISTANCE, args))

	fileop.write(config.C_DISTANCE, outdata)

def _run_feats_real ():
	"""Run distances with RealFeatures."""

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
	"""Run distances with complex StringFeatures, like WordString."""

	data=dataop.get_dna(len_seq_test_add=42)
	feats=featop.get_string_complex('Word', data)

	_compute('CanberraWordDistance', feats, data)
	_compute('HammingWordDistance', feats, data, False)
	_compute('HammingWordDistance', feats, data, True)
	_compute('ManhattanWordDistance', feats, data)

def run ():
	"""Run generator for all distances."""

	_run_feats_real()
	_run_feats_string_complex()

