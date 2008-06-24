"""
Test Distance
"""

from sg import sg
import util


def test (indata):
	if indata['name'].startswith('Sparse'):
		print "Sparse features not supported yet!"
		return False

	util.set_features(indata)
	util.convert_features_and_add_preproc(indata)
	util.set_distance(indata)

	dmatrix=sg('get_distance_matrix')
	dtrain=max(abs(indata['dm_train']-dmatrix).flat)

	sg('init_distance', 'TEST')
	dmatrix=sg('get_distance_matrix')
	dtest=max(abs(indata['dm_test']-dmatrix).flat)

	return util.check_accuracy(indata['accuracy'], dtrain=dtrain, dtest=dtest)

