def gaussian_shift ():
	print 'GaussianShift'
	from shogun.Features import RealFeatures
	from shogun.Kernel import GaussianShiftKernel

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	width=1.8
	max_shift=2
	shift_step=1

	kernel=GaussianShiftKernel(
		feats_train, feats_train, width, max_shift, shift_step)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	gaussian_shift()
