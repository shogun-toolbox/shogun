#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list = [[traindat,testdat,10,1.2,1.3],[traindat,testdat,10,1.2,1.3]]

def kernel_sigmoid_modular (train_fname=traindat,test_fname=testdat,size_cache=10,gamma=1.2,coef0=1.3):
	from modshogun import RealFeatures, SigmoidKernel, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	kernel=SigmoidKernel(feats_train, feats_train, size_cache, gamma, coef0)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Sigmoid')
	kernel_sigmoid_modular(*parameter_list[0])
