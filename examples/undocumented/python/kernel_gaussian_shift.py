#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list=[[traindat,testdat,1.8,2,1],[traindat,testdat,1.9,2,1]]

def kernel_gaussian_shift_modular (train_fname=traindat,test_fname=testdat,width=1.8,max_shift=2,shift_step=1):
	from modshogun import RealFeatures, GaussianShiftKernel, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	kernel=GaussianShiftKernel(feats_train, feats_train, width, max_shift, shift_step)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

	return km_train,km_test,kernel

if __name__=='__main__':
	print('GaussianShift')
	kernel_gaussian_shift_modular(*parameter_list[0])
