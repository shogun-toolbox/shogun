#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat,testdat,1000,10],[traindat,testdat,1000,10]]

def preprocessor_randomfouriergausspreproc_modular (fm_train_real=traindat,fm_test_real=testdat,kernelwidth=1,size_cache=10):
	from shogun.Kernel import GaussianKernel
	from shogun.Kernel import LinearKernel
	from shogun.Features import RealFeatures
	from shogun.Preprocessor import RandomFourierGaussPreproc

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	inputdim=feats_train.get_num_features()
	outputdim=inputdim*100 # take outputdim = inputdim*30 is an arbitrary choice, use more dimensions for a better approximation
	preproc=RandomFourierGaussPreproc()
	preproc.set_parameters(inputdim,outputdim,kernelwidth)
	preproc.init_randomcoefficients_from_scratch() # do not repeat this when computing testfeatures!!, instead get the coefficients from training run via get_randomcoefficients(...) and plug them in for the testing run  

	rfgauss_train=preproc.apply_to_dotfeatures_sparse_or_dense_with_real(feats_train)
	#rfgauss_test=preproc.apply_to_dotfeatures_sparse_or_dense_with_real(feats_test)


	#compute RFGAUSS approx for training kernel
	kernel_rfgauss_train_vs_train=LinearKernel(rfgauss_train, rfgauss_train)
	km_train_rfgauss=kernel_rfgauss_train_vs_train.get_kernel_matrix()

	#compute exact gaussian kernel for testing kernel
	kernel_exact_train_vs_train=GaussianKernel(feats_train, feats_train, kernelwidth, size_cache)
	km_train_exact=kernel_exact_train_vs_train.get_kernel_matrix()


	#compute relative difference
	diffs=abs(km_train_rfgauss/km_train_exact-1)
	v1=reduce(lambda x, y: x+y / float(92),diffs, 0)

	print "inputdim", inputdim, " outputdim", outputdim, " kernelwidth", kernelwidth
	print "mean relative difference between exact kernel and random fourier approximation"
	print sum(v1)/float(92)

	
	return kernel_rfgauss_train_vs_train,kernel_exact_train_vs_train

if __name__=='__main__':
	print('RandomFourierGaussPreproc')
	preprocessor_randomfouriergausspreproc_modular(*parameter_list[0])
