#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import double
import shogun as sg
lm=LoadMatrix()

traindat = double(lm.load_numbers('../data/fm_train_real.dat'))
testdat = double(lm.load_numbers('../data/fm_test_real.dat'))
traindna = lm.load_dna('../data/fm_train_dna.dat')
testdna = lm.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,traindna,testdna],[traindat,testdat,traindna,testdna]]
def kernel_combined (fm_train_real=traindat,fm_test_real=testdat,fm_train_dna=traindna,fm_test_dna=testdna ):

	kernel=sg.create("CombinedKernel")
	feats_train=sg.create("CombinedFeatures")
	feats_test=sg.create("CombinedFeatures")

	subkfeats_train=sg.create_features(fm_train_real)
	subkfeats_test=sg.create_features(fm_test_real)
	subkernel=sg.create("GaussianKernel", width=1.0)
	feats_train.add("feature_array", subkfeats_train)
	feats_test.add("feature_array", subkfeats_test)
	kernel.add("kernel_array", subkernel)

	subkfeats_train=sg.create_string_features(fm_train_dna, sg.DNA)
	subkfeats_test=sg.create_string_features(fm_test_dna, sg.DNA)
	degree=3
	subkernel=sg.create("FixedDegreeStringKernel", degree=degree)
	feats_train.add("feature_array", subkfeats_train)
	feats_test.add("feature_array", subkfeats_test)
	kernel.add("kernel_array", subkernel)

	subkfeats_train=sg.create_string_features(fm_train_dna, sg.DNA)
	subkfeats_test=sg.create_string_features(fm_test_dna, sg.DNA)
	subkernel=sg.create("LocalAlignmentStringKernel")
	feats_train.add("feature_array", subkfeats_train)
	feats_test.add("feature_array", subkfeats_test)
	kernel.add("kernel_array", subkernel)

	print(feats_train)
	print(feats_test)

	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('Combined')
	kernel_combined(*parameter_list[0])


