"""Generator for Distance"""

import shogun.Distance as distance

import fileop
import dataop
import featop
import category


def _compute (feats, params):
	"""Compute a distance and gather result data.

	@param feats Train and test features
	@param params dict with parameters to distance
	"""

	fun=eval('distance.'+params['name'])
	if params.has_key('args'):
		dist=fun(feats['train'], feats['train'], *params['args']['val'])
	else:
		dist=fun(feats['train'], feats['train'])
	dm_train=dist.get_distance_matrix()
	dist.init(feats['train'], feats['test'])
	dm_test=dist.get_distance_matrix()

	output={
		'distance_matrix_train':dm_train,
		'distance_matrix_test':dm_test,
	}
	output.update(fileop.get_output(category.DISTANCE, params))

	fileop.write(category.DISTANCE, output)

def _run_feats_real ():
	"""Run distances with RealFeatures."""

	params={
		'accuracy': 1e-8,
		'feature_class': 'simple',
		'feature_type': 'Real',
		'data': dataop.get_rand()
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])

	params['name']='EuclidianDistance'
	_compute(feats, params)
	params['name']='CanberraMetric'
	_compute(feats, params)
	params['name']='ChebyshewMetric'
	_compute(feats, params)
	params['name']='GeodesicMetric'
	_compute(feats, params)
	params['name']='JensenMetric'
	_compute(feats, params)
	params['name']='ManhattanMetric'
	_compute(feats, params)
	params['name']='BrayCurtisDistance'
	_compute(feats, params)
	params['name']='ChiSquareDistance'
	_compute(feats, params)
	params['name']='CosineDistance'
	_compute(feats, params)
	params['name']='TanimotoDistance'
	_compute(feats, params)
	params['name']='ManhattanMetric'
	_compute(feats, params)
	params['name']='MinkowskiMetric'
	params['args']={'key': ('k',), 'val': (1.3,)}
	_compute(feats, params)

	params['name']='SparseEuclidianDistance'
	params['accuracy']=1e-7
	del params['args']
	feats=featop.get_features(
		params['feature_class'], params['feature_type'],
		params['data'], sparse=True)
	_compute(feats, params)


def _run_feats_string_complex ():
	"""Run distances with complex StringFeatures, like WordString."""

	params={
		'accuracy': 1e-7,
		'feature_class': 'string_complex',
		'feature_type': 'Word',
		'data': dataop.get_dna(num_vec_test=dataop.NUM_VEC_TRAIN+42)
	}
	feats=featop.get_features(
		params['feature_class'], params['feature_type'], params['data'])

	params['name']='CanberraWordDistance'
	_compute(feats, params)

	params['accuracy']=1e-8
	params['name']='ManhattanWordDistance'
	_compute(feats, params)

	params['name']='HammingWordDistance'
	params['args']={'key': ('use_sign',), 'val': (False,)}
	_compute(feats, params)
	params['name']='HammingWordDistance'
	params['args']={'key': ('use_sign',), 'val': (True,)}
	_compute(feats, params)


def run ():
	"""Run generator for all distances."""

	_run_feats_real()
	_run_feats_string_complex()

