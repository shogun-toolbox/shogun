#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import double
lm=LoadMatrix()

traindat = double(lm.load_numbers('../data/fm_train_real.dat'))
testdat = double(lm.load_numbers('../data/fm_test_real.dat'))
traindna = lm.load_dna('../data/fm_train_dna.dat')
testdna = lm.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,traindna,testdna],[traindat,testdat,traindna,testdna]]
def kernel_combined (fm_train_real=traindat,fm_test_real=testdat,fm_train_dna=traindna,fm_test_dna=testdna ):
	from shogun import FixedDegreeStringKernel, LocalAlignmentStringKernel
	from shogun import StringCharFeatures, CombinedFeatures, DNA
	import shogun as sg

	kernel=sg.create_kernel("CombinedKernel")
	feats_train=CombinedFeatures()
	feats_test=CombinedFeatures()

	subkfeats_train=sg.create_features(fm_train_real)
	subkfeats_test=sg.create_features(fm_test_real)
	subkernel=sg.create_kernel("GaussianKernel", log_width=1.1)
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.add("kernel_array", subkernel)

	subkfeats_train=StringCharFeatures(fm_train_dna, DNA)
	subkfeats_test=StringCharFeatures(fm_test_dna, DNA)
	degree=3
	subkernel=FixedDegreeStringKernel(10, degree)
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.add("kernel_array", subkernel)

	subkfeats_train=StringCharFeatures(fm_train_dna, DNA)
	subkfeats_test=StringCharFeatures(fm_test_dna, DNA)
	subkernel=LocalAlignmentStringKernel(10)
	feats_train.append_feature_obj(subkfeats_train)
	feats_test.append_feature_obj(subkfeats_test)
	kernel.add("kernel_array", subkernel)

	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Combined')
	kernel_combined(*parameter_list[0])


