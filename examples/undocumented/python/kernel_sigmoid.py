#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list = [[traindat,testdat,10,1.2,1.3],[traindat,testdat,10,1.2,1.3]]

def kernel_sigmoid (train_fname=traindat,test_fname=testdat,size_cache=10,gamma=1.2,coef0=1.3):
	import shogun as sg

	feats_train=sg.features(sg.csv_file(train_fname))
	feats_test=sg.features(sg.csv_file(test_fname))

	kernel=sg.kernel("SigmoidKernel", gamma=gamma, coef0=coef0)
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Sigmoid')
	kernel_sigmoid(*parameter_list[0])
