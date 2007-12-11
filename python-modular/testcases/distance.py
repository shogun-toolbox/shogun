"""
Test Distance
"""

from shogun.Distance import *

import util

def _distance (indata, feats):
	fun=eval(indata['name'])
	args=util.get_args(indata, 'distance_arg')

	distance=fun(feats['train'], feats['train'], *args)
	dtrain=max(abs(indata['dm_train']-distance.get_distance_matrix()).flat)
	distance.init(feats['train'], feats['test'])
	dtest=max(abs(indata['dm_test']-distance.get_distance_matrix()).flat)

	return util.check_accuracy(indata['accuracy'], dtrain=dtrain, dtest=dtest)

def test (indata):
	fun=eval('util.get_feats_'+indata['feature_class'])
	feats=fun(indata)
	return _distance(indata, feats)

