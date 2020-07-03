#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list=[[traindat,testdat,1.2],[traindat,testdat,1.4]]

def kernel_linear (train_fname=traindat,test_fname=testdat,scale=1.2):
	import shogun as sg

	feats_train=sg.create_features(sg.read_csv(train_fname))
	feats_test=sg.create_features(sg.read_csv(test_fname))

	kernel=sg.create_kernel("LinearKernel")
	kernel.set_normalizer(sg.create_kernel_normalizer("AvgDiagKernelNormalizer", scale=scale))
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Linear')
	kernel_linear(*parameter_list[0])
