#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list=[[traindat,testdat,1.2],[traindat,testdat,1.4]]

def kernel_linear (train_fname=traindat,test_fname=testdat,scale=1.2):
	import shogun as sg
	from shogun import AvgDiagKernelNormalizer

	feats_train=sg.features(sg.csv_file(train_fname))
	feats_test=sg.features(sg.csv_file(test_fname))

	kernel=sg.kernel("LinearKernel")
	kernel.set_normalizer(AvgDiagKernelNormalizer(scale))
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Linear')
	kernel_linear(*parameter_list[0])
