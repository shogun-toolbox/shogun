#!/usr/bin/env python
import shogun as sg

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
parameter_list=[[traindat,testdat, 1.0],[traindat,testdat, 5.0]]

def kernel_rationalquadratic (train_fname=traindat,test_fname=testdat, shift_coef=1.0):
	from shogun import RealFeatures, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	distance = sg.distance('EuclideanDistance')

	kernel = sg.kernel('RationalQuadraticKernel', coef=shift_coef,
			   distance=distance)
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('RationalQuadratic')
	kernel_rationalquadratic(*parameter_list[0])
