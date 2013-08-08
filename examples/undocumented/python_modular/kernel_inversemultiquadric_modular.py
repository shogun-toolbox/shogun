#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list=[[traindat,testdat, 1.0],[traindat,testdat, 5.0]]

def kernel_inversemultiquadric_modular (train_fname=traindat,test_fname=testdat, shift_coef=1.0):
	from modshogun import RealFeatures, InverseMultiQuadricKernel, EuclideanDistance, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))
	
	distance=EuclideanDistance(feats_train, feats_train)
	kernel=InverseMultiQuadricKernel(feats_train, feats_train, shift_coef, distance)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('InverseMultiquadric')
	kernel_inversemultiquadric_modular(*parameter_list[0])
