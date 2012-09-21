"""
Test Distance
"""

from sg import sg
import util


def _evaluate (indata, prefix):
	dmatrix=sg('get_distance_matrix', 'TRAIN')
	dm_train=max(abs(indata['distance_matrix_train']-dmatrix).flat)

	dmatrix=sg('get_distance_matrix', 'TEST')
	dm_test=max(abs(indata['distance_matrix_test']-dmatrix).flat)

	return util.check_accuracy(
		indata[prefix+'accuracy'], dm_train=dm_train, dm_test=dm_test)


########################################################################
# public
########################################################################

def test (indata):
	prefix='distance_'
	try:
		util.set_features(indata, prefix)
	except NotImplementedError, e:
		print e
		return True

	util.set_and_train_distance(indata)

	return _evaluate(indata, prefix)

