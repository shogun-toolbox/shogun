#!/usr/bin/env python
import shogun as sg
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list=[[traindat,testdat, 1.0],[traindat,testdat, 5.0]]

def kernel_circular(train_fname=traindat,test_fname=testdat, sigma=1.0):
	from shogun import RealFeatures, distance, kernel, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	distance = sg.distance('EuclideanDistance')
	kernel = sg.kernel('CircularKernel', sigma=sigma, distance=distance)
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('Circular')
	kernel_circular(*parameter_list[0])
