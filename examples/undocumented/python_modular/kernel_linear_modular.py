from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
parameter_list=[[traindat,testdat,1.2],[traindat,testdat,1.4]]

def kernel_linear_modular (fm_train_real=traindat,fm_test_real=testdat,scale=1.2):

	from shogun.Features import RealFeatures
	from shogun.Kernel import LinearKernel, AvgDiagKernelNormalizer

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	 
	kernel=LinearKernel()
	kernel.set_normalizer(AvgDiagKernelNormalizer(scale))
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Linear')
	kernel_linear_modular(*parameter_list[0])
