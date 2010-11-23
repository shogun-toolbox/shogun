###########################################################################
# chi2 kernel
###########################################################################
from tools.load import LoadMatrix
from numpy import double
lm=LoadMatrix()

parameter_list = [[double(lm.load_numbers('../data/fm_train_real.dat')),double(lm.load_numbers('../data/fm_test_real.dat')),1.4,10], [double(lm.load_numbers('../data/fm_train_real.dat')),double(lm.load_numbers('../data/fm_test_real.dat')),1.5,10]]




def kernel_chi2_modular (fm_train_real=double(lm.load_numbers('../data/fm_train_real.dat')),fm_test_real=double(lm.load_numbers('../data/fm_test_real.dat')),width=1.4, size_cache=10):
	print 'Chi2'
	from shogun.Kernel import Chi2Kernel
	from shogun.Features import RealFeatures
	fm_train_real    = fm_train_real
	fm_test_real     = fm_test_real
	width            = width
	size_cache       = size_cache
	
	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	
	
	kernel=Chi2Kernel(feats_train, feats_train, width, size_cache)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	print km_train
	print km_test

if __name__=='__main__':
	from tools.load import LoadMatrix
	from numpy import double
	lm=LoadMatrix()
	fm_train_real=double(lm.load_numbers('../data/fm_train_real.dat'))
	fm_test_real=double(lm.load_numbers('../data/fm_test_real.dat'))
	kernel_chi2_modular(*parameter_list[0])
