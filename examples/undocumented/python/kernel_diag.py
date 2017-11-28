#!/usr/bin/env python
parameter_list =[[23],[24]]
def kernel_diag (diag=23):
	from shogun import DummyFeatures
	from shogun import DiagKernel

	feats_train=DummyFeatures(10)
	feats_test=DummyFeatures(17)

	kernel=DiagKernel(feats_train, feats_train, diag)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Diag')
	kernel_diag(*parameter_list[0])

