#!/usr/bin/env python
import shogun as sg

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
parameter_list=[[traindat,testdat, 1.0],[traindat,testdat, 5.0]]

def kernel_rationalquadratic (train_fname=traindat,test_fname=testdat, shift_coef=1.0):

	feats_train=sg.create_features(sg.read_csv(train_fname))
	feats_test=sg.create_features(sg.read_csv(test_fname))

	distance = sg.create_distance('EuclideanDistance')

	kernel = sg.create_kernel('RationalQuadraticKernel', coef=shift_coef,
			   distance=distance)
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel


if __name__=='__main__':
	print('RationalQuadratic')
	kernel_rationalquadratic(*parameter_list[0])
