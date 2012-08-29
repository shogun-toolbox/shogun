#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat,1.5,10],[traindat,testdat,1.5,10]]

def preprocessor_randomfouriergausspreproc_modular (fm_train_real=traindat,fm_test_real=testdat,width=1.4,size_cache=10):
	from shogun.Kernel import Chi2Kernel
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import RandomFourierGaussPreproc

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	preproc=RandomFourierGaussPreproc()
	preproc.init(feats_train)
	feats_train.add_preprocessor(preproc)
	feats_train.apply_preprocessor()
	feats_test.add_preprocessor(preproc)
	feats_test.apply_preprocessor()

	kernel=Chi2Kernel(feats_train, feats_train, width, size_cache)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

	return km_train,km_test,kernel

if __name__=='__main__':
	print('RandomFourierGaussPreproc')
	preprocessor_randomfouriergausspreproc_modular(*parameter_list[0])
