#!/usr/bin/env python
traindat = '../data/fm_train_byte.dat'
testdat = '../data/fm_test_byte.dat'

parameter_list=[[traindat,testdat],[traindat,testdat]]

def kernel_linear_byte_modular (train_fname=traindat,test_fname=testdat):
	from modshogun import LinearKernel, ByteFeatures, CSVFile

	feats_train=ByteFeatures(CSVFile(train_fname))
	feats_test=ByteFeatures(CSVFile(test_fname))

	kernel=LinearKernel(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return kernel

if __name__=='__main__':
	print('LinearByte')
	kernel_linear_byte_modular(*parameter_list[0])
