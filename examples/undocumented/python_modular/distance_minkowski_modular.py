#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list = [[traindat,testdat,3],[traindat,testdat,4]]

def distance_minkowski_modular (train_fname=traindat,test_fname=testdat,k=3):
	from modshogun import RealFeatures, MinkowskiMetric, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	distance=MinkowskiMetric(feats_train, feats_train, k)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

	return distance,dm_train,dm_test

if __name__=='__main__':
	print('MinkowskiMetric')
	distance_minkowski_modular(*parameter_list[0])

