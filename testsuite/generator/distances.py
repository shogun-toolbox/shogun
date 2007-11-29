from numpy import *
from shogun.Distance import *

import fileops

import dataops
import featops


def _get_output (name, output, args=[]):
	return output

def _compute (name, feats, data, *args):
	dfun=eval(name+'Distance')
	d=dfun(feats['train'], feats['train'], *args)
	dm_train=d.get_distance_matrix()
	d.init(feats['train'], feats['test'])
	dm_test=d.get_distance_matrix()

	output=_get_output(name, {
		'dm_train':dm_train,
		'dm_test':dm_test,
		'data_train':matrix(data['train']),
		'data_test':matrix(data['test'])
	}, args)

	return [name, output]


def _run_feats_real ():
	data=dataops.get_rand()
	feats=featops.get_simple('Real', data)

	fileops.write(_compute('NormSquared', feats, data))

def run ():
	fileops.TYPE='Distance'
	_run_feats_real()

