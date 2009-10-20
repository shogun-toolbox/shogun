def sparse_gaussian ():
	print 'SparseGaussian'
	from shogun.Features import SparseRealFeatures
	from shogun.Kernel import SparseGaussianKernel

	feats_train=SparseRealFeatures(fm_train_real)
	feats_test=SparseRealFeatures(fm_test_real)

	width=1.1
	kernel=SparseGaussianKernel(feats_train, feats_train, width)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()


if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	sparse_gaussian()
