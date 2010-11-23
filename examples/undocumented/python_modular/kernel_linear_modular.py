from tools.load import LoadMatrix
lm=LoadMatrix()
parameter_list=[[lm.load_numbers('../data/fm_train_real.dat'),lm.load_numbers('../data/fm_test_real.dat'),1.2],[lm.load_numbers('../data/fm_train_real.dat'),lm.load_numbers('../data/fm_test_real.dat'),1.4]]

def kernel_linear_modular (fm_train_real=lm.load_numbers('../data/fm_train_real.dat'),fm_test_real=lm.load_numbers('../data/fm_test_real.dat'),scale=1.2):
	print 'Linear'
	from shogun.Features import RealFeatures
	from shogun.Kernel import LinearKernel, AvgDiagKernelNormalizer
	fm_train_real=fm_train_real
	fm_test_real = fm_test_real
	scale = scale
	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	 
	kernel=LinearKernel()
	kernel.set_normalizer(AvgDiagKernelNormalizer(scale))
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	print km_train
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	print km_test

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	kernel_linear_modular(*parameter_list[0])
