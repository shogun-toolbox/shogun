#!/usr/bin/env python
parameter_list =[[23],[24]]
def kernel_diag (diag=23):
	from shogun import DummyFeatures
	import shogun as sg

	feats_train=DummyFeatures(10)
	feats_test=DummyFeatures(17)

	kernel=sg.kernel("DiagKernel", diag=diag)
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Diag')
	kernel_diag(*parameter_list[0])

