from numpy import *
from shogun.Distance import *

import fileops
import dataops
import featops

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
	output.update(fileops.get_output_params(name, args))

	return [name, output]

def _run_feats_real ():
	data=dataops.get_rand()
	feats=featops.get_simple('Real', data)

	fileops.write(_compute('EuclidianDistance', feats, data))

	feats=featops.get_simple('Real', data, sparse=True)
	fileops.write(_compute('SparseEuclidianDistance', feats, data))

def _run_feats_string_complex ():
	data=dataops.get_dna()
	feats=featops.get_string_complex('Word', data)

	fileops.write(_compute('CanberraWordDistance', feats, data))
	fileops.write(_compute('HammingWordDistance', feats, data, False))
	fileops.write(_compute('HammingWordDistance', feats, data, True))
	fileops.write(_compute('ManhattanWordDistance', feats, data))

def run ():
	fileops.TYPE='Distance'

	_run_feats_real()
	_run_feats_string_complex()

