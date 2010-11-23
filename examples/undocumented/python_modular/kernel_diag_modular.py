parameter_list =[[23],[24]]
def kernel_diag_modular (diag=23):
	print 'Diag'
	from shogun.Features import DummyFeatures
	from shogun.Kernel import DiagKernel

	feats_train=DummyFeatures(10)
	feats_test=DummyFeatures(17)
	diag=diag

	kernel=DiagKernel(feats_train, feats_train, diag)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	print km_test
if __name__=='__main__':
	kernel_diag_modular(*parameter_list[0])
