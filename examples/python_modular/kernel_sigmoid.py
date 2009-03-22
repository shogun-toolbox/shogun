def sigmoid ():
	print 'Sigmoid'
	from shogun.Features import RealFeatures
	from shogun.Kernel import SigmoidKernel

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	size_cache=10
	gamma=1.2
	coef0=1.3

	kernel=SigmoidKernel(feats_train, feats_train, size_cache, gamma, coef0)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	sigmoid()
