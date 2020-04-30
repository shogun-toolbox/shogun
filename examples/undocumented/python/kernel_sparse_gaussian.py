#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()
traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat,1.1],[traindat,testdat,1.2]]

def kernel_sparse_gaussian (fm_train_real=traindat,fm_test_real=testdat,width=1.1 ):
	from shogun import SparseRealFeatures
	import shogun as sg

	feats_train=SparseRealFeatures(fm_train_real)
	feats_test=SparseRealFeatures(fm_test_real)

	kernel=sg.create_kernel("GaussianKernel", log_width=width)
	kernel.init(feats_train, feats_train,)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('SparseGaussian')
	kernel_sparse_gaussian (*parameter_list[0])
