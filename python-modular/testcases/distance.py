from shogun.Distance import *

import util

def _distance (input, feats):
	dfun=eval(input['name'])
	args=util.get_args(input, 'distance_arg')

	d=dfun(feats['train'], feats['train'], *args)
	train=max(abs(input['dm_train']-d.get_distance_matrix()).flat)
	d.init(feats['train'], feats['test'])
	test=max(abs(input['dm_test']-d.get_distance_matrix()).flat)

	return util.check_accuracy(input['accuracy'], train=train, test=test)

def test (input):
	fun=eval('util.get_feats_'+input['feature_class'])
	feats=fun(input)
	return _distance(input, feats)

