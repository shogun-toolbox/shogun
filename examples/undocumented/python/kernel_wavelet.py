#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import where

lm=LoadMatrix()
traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat, 1.5, 1.0],[traindat,testdat, 1.0, 1.5]]

def kernel_wavelet (fm_train_real=traindat,fm_test_real=testdat, dilation=1.5, translation=1.0):
	import shogun as sg

	feats_train=sg.features(fm_train_real)
	feats_test=sg.features(fm_test_real)

	kernel=sg.kernel("WaveletKernel", dilation=dilation, translation=translation)
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('Wavelet')
	kernel_wavelet(*parameter_list[0])
