#!/usr/bin/env python
import shogun as sg
testdat = '../data/fm_train_real.dat'
traindat = '../data/fm_test_real.dat'

parameter_list=[[traindat,testdat,1.7],[traindat,testdat,1.8]]

def kernel_distance (train_fname=traindat,test_fname=testdat,width=1.7):
	from shogun import RealFeatures, distance, kernel, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	distance = sg.distance('EuclideanDistance')
	kernel = sg.kernel('DistanceKernel', width=width, distance=distance)
	kernel.init(feats_train, feats_test)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Distance')
	kernel_distance(*parameter_list[0])
