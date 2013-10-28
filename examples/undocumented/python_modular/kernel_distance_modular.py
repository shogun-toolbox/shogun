#!/usr/bin/env python
testdat = '../data/fm_train_real.dat'
traindat = '../data/fm_test_real.dat'

parameter_list=[[traindat,testdat,1.7],[traindat,testdat,1.8]]

def kernel_distance_modular (train_fname=traindat,test_fname=testdat,width=1.7):
	from modshogun import RealFeatures, DistanceKernel, EuclideanDistance, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	distance=EuclideanDistance()
	kernel=DistanceKernel(feats_train, feats_test, width, distance)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Distance')
	kernel_distance_modular(*parameter_list[0])
