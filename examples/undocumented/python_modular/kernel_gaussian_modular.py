#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list=[[traindat,testdat, 1.3],[traindat,testdat, 1.4]]

def kernel_gaussian_modular (train_fname=traindat,test_fname=testdat, width=1.3):
	from modshogun import RealFeatures, GaussianKernel, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	kernel=GaussianKernel(feats_train, feats_train, width)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('Gaussian')
	kernel_gaussian_modular(*parameter_list[0])
