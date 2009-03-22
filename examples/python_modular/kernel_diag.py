def diag ():
	print 'Diag'
	from shogun.Features import DummyFeatures
	from shogun.Kernel import DiagKernel

	feats_train=DummyFeatures(10)
	feats_test=DummyFeatures(17)
	diag=23.

	kernel=DiagKernel(feats_train, feats_train, diag)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

if __name__=='__main__':
	diag()
