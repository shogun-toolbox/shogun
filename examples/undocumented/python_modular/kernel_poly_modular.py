#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()
traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat,4,False,True],[traindat,testdat,5,False,True]]

def kernel_poly_modular (fm_train_real=traindat,fm_test_real=testdat,degree=4,inhomogene=False,
	use_normalization=True):
	from modshogun import RealFeatures
	from modshogun import PolyKernel

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	kernel=PolyKernel(
		feats_train, feats_train, degree, inhomogene, use_normalization)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel
if __name__=='__main__':
	print('Poly')
	kernel_poly_modular (*parameter_list[0])
