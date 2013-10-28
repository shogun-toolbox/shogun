#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list = [[traindat,testdat,4,False,True],[traindat,testdat,5,False,True]]

def kernel_poly_modular (train_fname=traindat,test_fname=testdat,degree=4,inhomogene=False,
	use_normalization=True):
	from modshogun import RealFeatures, PolyKernel, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	kernel=PolyKernel(
		feats_train, feats_train, degree, inhomogene, use_normalization)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel
if __name__=='__main__':
	print('Poly')
	kernel_poly_modular (*parameter_list[0])
