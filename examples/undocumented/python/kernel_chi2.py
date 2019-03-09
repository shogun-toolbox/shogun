#!/usr/bin/env python
traindat = '../data/fm_train_hist.dat'
testdat = '../data/fm_test_hist.dat'

parameter_list = [[traindat,testdat,1.4,10], [traindat,testdat,1.5,10]]

def kernel_chi2 (train_fname=traindat,test_fname=testdat,width=1.4, size_cache=10):
	from shogun import RealFeatures, Chi2Kernel, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	kernel=Chi2Kernel(feats_train, feats_train, width, size_cache)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Chi2')
	kernel_chi2(*parameter_list[0])
