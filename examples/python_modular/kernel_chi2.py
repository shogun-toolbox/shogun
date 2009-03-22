###########################################################################
# chi2 kernel
###########################################################################
def chi2 ():
	print 'Chi2'
	from shogun.Kernel import Chi2Kernel
	from shogun.Features import RealFeatures

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	width=1.4
	size_cache=10
	
	kernel=Chi2Kernel(feats_train, feats_train, width, size_cache)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

if __name__=='__main__':
	from tools.load import LoadMatrix
	from numpy import double
	lm=LoadMatrix()
	fm_train_real=double(lm.load_numbers('../data/fm_train_real.dat'))
	fm_test_real=double(lm.load_numbers('../data/fm_test_real.dat'))
	chi2()
