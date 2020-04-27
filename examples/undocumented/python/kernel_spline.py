#!/usr/bin/env python
from tools.load import LoadMatrix

lm=LoadMatrix()
traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat],[traindat,testdat]]

def kernel_spline (fm_train_real=traindat,fm_test_real=testdat):
	import shogun as sg

	feats_train=sg.create_features(fm_train_real)
	feats_test=sg.create_features(fm_test_real)

	kernel=sg.create_kernel("SplineKernel")
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('Spline')
	kernel_spline(*parameter_list[0])
