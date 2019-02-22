#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import where
import shogun as sg

lm=LoadMatrix()
traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list=[[traindat,testdat, 2.0],[traindat,testdat, 3.0]]

def kernel_tstudent (fm_train_real=traindat,fm_test_real=testdat, degree=2.0):
	from shogun import RealFeatures
	from shogun import kernel, distance

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance = sg.distance('EuclideanDistance')

	kernel = sg.kernel('TStudentKernel', degree=degree, distance=distance)
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('TStudent')
	kernel_tstudent(*parameter_list[0])
