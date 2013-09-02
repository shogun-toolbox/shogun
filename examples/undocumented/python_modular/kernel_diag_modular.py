#!/usr/bin/env python
parameter_list =[[23],[24]]
def kernel_diag_modular (diag=23):
	from modshogun import DummyFeatures
	from modshogun import DiagKernel

	feats_train=DummyFeatures(10)
	feats_test=DummyFeatures(17)

	kernel=DiagKernel(feats_train, feats_train, diag)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Diag')
	kernel_diag_modular(*parameter_list[0])
	
