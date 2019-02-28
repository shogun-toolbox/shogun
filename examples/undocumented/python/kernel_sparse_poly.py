#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()
traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat,10,3,1.0],[traindat,testdat,10,4,1.0]]

def kernel_sparse_poly (fm_train_real=traindat,fm_test_real=testdat,
						cache_size=10,degree=3,c=1.0):

	from shogun import SparseRealFeatures
	import shogun as sg

	feats_train=SparseRealFeatures(fm_train_real)
	feats_test=SparseRealFeatures(fm_test_real)



	kernel=sg.kernel("PolyKernel", cache_size=cache_size, degree=degree,
					 c=c)
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('SparsePoly')
	kernel_sparse_poly(*parameter_list[0])
