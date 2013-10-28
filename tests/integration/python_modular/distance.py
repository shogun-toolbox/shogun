"""
Test Distance
"""

from modshogun import *

import util


def _evaluate (indata):
	prefix='distance_'
	feats=util.get_features(indata, prefix)

	dfun=eval(indata[prefix+'name'])
	dargs=util.get_args(indata, prefix)
	distance=dfun(feats['train'], feats['train'], *dargs)

	dm_train=max(abs(
		indata[prefix+'matrix_train']-distance.get_distance_matrix()).flat)
	distance.init(feats['train'], feats['test'])
	dm_test=max(abs(
		indata[prefix+'matrix_test']-distance.get_distance_matrix()).flat)

	return util.check_accuracy(
		indata[prefix+'accuracy'], dm_train=dm_train, dm_test=dm_test)


########################################################################
# public
########################################################################

def test (indata):
	return _evaluate(indata)

