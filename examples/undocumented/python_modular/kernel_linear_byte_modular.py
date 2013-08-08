#!/usr/bin/env python
from numpy import ubyte
lm=LoadMatrix()

traindat = ubyte(lm.load_numbers('../data/fm_train_byte.dat'))
testdat = ubyte(lm.load_numbers('../data/fm_test_byte.dat'))

parameter_list=[[traindat,testdat],[traindat,testdat]]

def kernel_linear_byte_modular (fm_train_byte=traindat,fm_test_byte=testdat):
	from shogun.Kernel import LinearKernel
	from shogun.Features import ByteFeatures

	feats_train=ByteFeatures(fm_train_byte)
	feats_test=ByteFeatures(fm_test_byte)

	kernel=LinearKernel(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return kernel

if __name__=='__main__':
	print('LinearByte')
	kernel_linear_byte_modular(*parameter_list[0])
