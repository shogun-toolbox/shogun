#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()
traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat,10,3,True],[traindat,testdat,10,4,True]]

def kernel_sparse_poly_modular (fm_train_real=traindat,fm_test_real=testdat,
		 size_cache=10,degree=3,inhomogene=True ):

	from modshogun import SparseRealFeatures
	from modshogun import PolyKernel

	feats_train=SparseRealFeatures(fm_train_real)
	feats_test=SparseRealFeatures(fm_test_real)



	kernel=PolyKernel(feats_train, feats_train, size_cache,
		inhomogene, degree)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('SparsePoly')
	kernel_sparse_poly_modular(*parameter_list[0])
