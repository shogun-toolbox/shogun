#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
parameter_list = [[traindat,testdat,2,10], [traindat,testdat,5,10]]

def kernel_anova_modular (train_fname=traindat,test_fname=testdat,cardinality=2, size_cache=10):
	from modshogun import ANOVAKernel,RealFeatures,CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	kernel=ANOVAKernel(feats_train, feats_train, cardinality, size_cache)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train, km_test, kernel

if __name__=='__main__':
	print('ANOVA')
	kernel_anova_modular(*parameter_list[0])
