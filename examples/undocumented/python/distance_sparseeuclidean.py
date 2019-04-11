#!/usr/bin/env python
import shogun as sg
traindat = '../data/fm_train_sparsereal.dat'
testdat = '../data/fm_test_sparsereal.dat'

parameter_list = [[traindat,testdat],[traindat,testdat]]

def distance_sparseeuclidean (train_fname=traindat,test_fname=testdat):
	from shogun import SparseRealFeatures
	import shogun as sg

	feats_train=SparseRealFeatures(sg.libsvm_file(train_fname))
	feats_test=SparseRealFeatures(sg.libsvm_file(test_fname))
	
	print feats_train.get_num_features()
	print feats_test.get_num_features()

	distance = sg.distance('SparseEuclideanDistance')
	distance.init(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

	return distance,dm_train,dm_test

if __name__=='__main__':
	print('SparseEuclideanDistance')
	distance_sparseeuclidean(*parameter_list[0])
