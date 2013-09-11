#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list = [[traindat, testdat]]

def distance_mahalanobis_modular (train_fname = traindat, test_fname = testdat):

	from modshogun import RealFeatures
	from modshogun import MahalanobisDistance

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	distance = MahalanobisDistance(feats_train, feats_train)
	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

	return distance,dm_train,dm_test

if __name__=='__main__':
	print('MahalanobisDistance')
	distance_mahalanobis_modular(*parameter_list[0])
