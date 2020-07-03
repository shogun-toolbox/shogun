#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()
traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,5,5],[traindat,testdat,6,6]]

def kernel_distantsegments (fm_train_dna=traindat,fm_test_dna=testdat,delta=5, theta=5):
	from shogun import DistantSegmentsKernel

	feats_train=sg.create_string_features(fm_train_dna, sg.DNA)
	feats_test=sg.create_string_features(fm_test_dna, sg.DNA)

	kernel=DistantSegmentsKernel(feats_train, feats_train, 10, delta, theta)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

	return km_train, km_test, kernel


if __name__=='__main__':
	print('DistantSegments')
	kernel_distantsegments(*parameter_list[0])
