#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list = [[traindat,testdat],[traindat,testdat]]

def distance_sparseeuclidean (train_fname=traindat,test_fname=testdat):
	from shogun import RealFeatures, SparseRealFeatures, SparseEuclideanDistance, CSVFile

	realfeat=RealFeatures(CSVFile(train_fname))
	feats_train=SparseRealFeatures(realfeat)
	realfeat=RealFeatures(CSVFile(test_fname))
	feats_test=SparseRealFeatures(realfeat)

	distance=SparseEuclideanDistance(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

	return distance,dm_train,dm_test

if __name__=='__main__':
	print('SparseEuclideanDistance')
	distance_sparseeuclidean(*parameter_list[0])
