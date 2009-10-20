def const ():
	print 'Const'
	from shogun.Features import DummyFeatures
	from shogun.Kernel import ConstKernel

	feats_train=DummyFeatures(10)
	feats_test=DummyFeatures(17)
	c=23.

	kernel=ConstKernel(feats_train, feats_train, c)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()


if __name__=='__main__':
	const()
