#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import where

lm=LoadMatrix()
traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat, 1.0],[traindat,testdat, 10.0]]

def kernel_wave_modular (fm_train_real=traindat,fm_test_real=testdat, theta=1.0):
	from modshogun import RealFeatures
	from modshogun import WaveKernel
	from modshogun import EuclideanDistance

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	
	distance=EuclideanDistance(feats_train, feats_train)

	kernel=WaveKernel(feats_train, feats_train, theta, distance)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('Wave')
	kernel_wave_modular(*parameter_list[0])
