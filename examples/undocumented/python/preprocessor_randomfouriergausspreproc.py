#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat,1.5,10],[traindat,testdat,1.5,10]]

from shogun import Math_init_random;
Math_init_random(12345);

def preprocessor_randomfouriergausspreproc (fm_train_real=traindat,fm_test_real=testdat,width=1.4,size_cache=10):
	from shogun import Chi2Kernel
	from shogun import RealFeatures
	from shogun import RandomFourierGaussPreproc

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	preproc=RandomFourierGaussPreproc()
	preproc.fit(feats_train)
	feats_train = preproc.apply(feats_train)
	feats_test = preproc.apply(feats_test)

	kernel=Chi2Kernel(feats_train, feats_train, width, size_cache)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

	return km_train,km_test,kernel

if __name__=='__main__':
	print('RandomFourierGaussPreproc')
	preprocessor_randomfouriergausspreproc(*parameter_list[0])
