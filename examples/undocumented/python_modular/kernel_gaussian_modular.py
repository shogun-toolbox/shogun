from tools.load import LoadMatrix
from numpy import where
lm=LoadMatrix()

parameter_list=[[lm.load_numbers('../data/fm_train_real.dat'),lm.load_numbers('../data/fm_test_real.dat'),1.3],[lm.load_numbers('../data/fm_train_real.dat'),lm.load_numbers('../data/fm_test_real.dat'),1.4]]

def kernel_gaussian_modular (fm_train_real=lm.load_numbers('../data/fm_train_real.dat'),fm_test_real=lm.load_numbers('../data/fm_test_real.dat'),width=1.3):
	print 'Gaussian'
	from shogun.Features import RealFeatures
	from shogun.Kernel import GaussianKernel
	fm_train_real = fm_train_real
	fm_test_real = fm_test_real
	width = width

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	kernel=GaussianKernel(feats_train, feats_train, width)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	print km_train, km_test
#	return km_train,km_test


if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	kernel_gaussian_modular(*parameter_list[0])
