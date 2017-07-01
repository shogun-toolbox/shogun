#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list=[[traindat,testdat, 1.0],[traindat,testdat, 5.0]]

def kernel_exponential_modular (train_fname=traindat,test_fname=testdat, tau_coef=1.0):
	from modshogun import RealFeatures, ExponentialKernel, EuclideanDistance, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	distance = EuclideanDistance(feats_train, feats_train)
	kernel=ExponentialKernel(feats_train, feats_train, tau_coef, distance, 10)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('Exponential')
	kernel_exponential_modular(*parameter_list[0])
