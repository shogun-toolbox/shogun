#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import where
import shogun as sg

lm=LoadMatrix()
traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat, 1.0],[traindat,testdat, 10.0]]

def kernel_wave (fm_train_real=traindat,fm_test_real=testdat, theta=1.0):
	feats_train=sg.features(fm_train_real)
	feats_test=sg.features(fm_test_real)

	distance = sg.distance('EuclideanDistance')

	kernel = sg.kernel('WaveKernel', theta=theta, distance=distance)
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('Wave')
	kernel_wave(*parameter_list[0])
