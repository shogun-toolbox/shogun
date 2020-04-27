#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'

parameter_list = [[traindat,testdat,4,0.0],[traindat,testdat,5,0.0]]

def kernel_poly (train_fname=traindat,test_fname=testdat,degree=4,c=0.0):
	from shogun import CSVFile
	import shogun as sg

	feats_train=sg.create_features(CSVFile(train_fname))
	feats_test=sg.create_features(CSVFile(test_fname))

	kernel = sg.create_kernel("PolyKernel", degree=degree, c=c)
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel
if __name__=='__main__':
	print('Poly')
	kernel_poly (*parameter_list[0])
