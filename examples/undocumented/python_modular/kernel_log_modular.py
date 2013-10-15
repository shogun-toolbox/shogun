#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list=[[traindat,testdat, 2.0],[traindat,testdat, 3.0]]

def kernel_log_modular (train_fname=traindat,test_fname=testdat, degree=2.0):
	from modshogun import RealFeatures, LogKernel, EuclideanDistance, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	distance=EuclideanDistance(feats_train, feats_train)

	kernel=LogKernel(feats_train, feats_train, degree, distance)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('Log')
	kernel_log_modular(*parameter_list[0])
