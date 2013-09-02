#!/usr/bin/env python
from tools.load import LoadMatrix
lm = LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat  = lm.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat, testdat]]

def distance_mahalanobis_modular (fm_train_real = traindat, fm_test_real = testdat):

	from modshogun import RealFeatures
	from modshogun import MahalanobisDistance

	feats_train = RealFeatures(fm_train_real)
	feats_test  = RealFeatures(fm_test_real)

	distance = MahalanobisDistance(feats_train, feats_train)
	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

	return distance,dm_train,dm_test

if __name__=='__main__':
	print('MahalanobisDistance')
	distance_mahalanobis_modular(*parameter_list[0])
