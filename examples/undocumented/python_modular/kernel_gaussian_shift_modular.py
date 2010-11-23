from tools.load import LoadMatrix
lm=LoadMatrix()
parameter_list=[[lm.load_numbers('../data/fm_train_real.dat'),lm.load_numbers('../data/fm_test_real.dat'),1.8,2,1],[lm.load_numbers('../data/fm_train_real.dat'),lm.load_numbers('../data/fm_test_real.dat'),1.9,2,1]]

def kernel_gaussian_shift_modular (fm_train_real=lm.load_numbers('../data/fm_train_real.dat'),fm_test_real=lm.load_numbers('../data/fm_test_real.dat'),width=1.8,max_shift=2,shift_step=1):
	print 'GaussianShift'
	from shogun.Features import RealFeatures
	from shogun.Kernel import GaussianShiftKernel
	fm_train_real   =fm_train_real
	fm_test_real    =fm_test_real
	width           =width
	max_shift       =max_shift
	shift_step=shift_step
	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	kernel=GaussianShiftKernel(
		feats_train, feats_train, width, max_shift, shift_step)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	print km_train
        print km_test

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	kernel_gaussian_shift_modular(*parameter_list[0])
