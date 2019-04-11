#!/usr/bin/env python
traindat = '../data/fm_train_byte.dat'
testdat = '../data/fm_test_byte.dat'

parameter_list=[[traindat,testdat],[traindat,testdat]]

def kernel_linear_byte (train_fname=traindat,test_fname=testdat):
	from shogun import LinearKernel
	import shogun as sg

	feats_train=sg.features(sg.csv_file(train_fname), sg.PT_UINT8)
	feats_test=sg.features(sg.csv_file(test_fname), sg.PT_UINT8)

	kernel=sg.kernel("LinearKernel")
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return kernel

if __name__=='__main__':
	print('LinearByte')
	kernel_linear_byte(*parameter_list[0])
