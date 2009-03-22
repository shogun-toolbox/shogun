def poly ():
	print 'Poly'
	from shogun.Features import RealFeatures
	from shogun.Kernel import PolyKernel

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	degree=4
	inhomogene=False
	use_normalization=True
	
	kernel=PolyKernel(
		feats_train, feats_train, degree, inhomogene, use_normalization)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()


if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	poly()
