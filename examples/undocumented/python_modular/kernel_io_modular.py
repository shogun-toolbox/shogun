#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list=[[traindat,testdat,1.9],[traindat,testdat,1.7]]

def kernel_io_modular (train_fname=traindat,test_fname=testdat,width=1.9):
	from modshogun import RealFeatures, GaussianKernel, CSVFile
	
	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	kernel=GaussianKernel(feats_train, feats_train, width)
	km_train=kernel.get_kernel_matrix()
	f=CSVFile("gaussian_train.csv","w")
	kernel.save(f)
	del f

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	f=CSVFile("gaussian_test.csv","w")
	kernel.save(f)
	del f

	#clean up
	import os
	os.unlink("gaussian_test.csv")
	os.unlink("gaussian_train.csv")
	
	return km_train, km_test, kernel

if __name__=='__main__':
	print('Gaussian')
	kernel_io_modular(*parameter_list[0])
