#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,20],[traindat,testdat,22]]
def kernel_weighted_degree_position_string_modular (fm_train_dna=traindat,fm_test_dna=testdat,degree=20):
	from modshogun import StringCharFeatures, DNA
	from modshogun import WeightedDegreePositionStringKernel, MSG_DEBUG

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	#feats_train.io.set_loglevel(MSG_DEBUG)
	feats_test=StringCharFeatures(fm_test_dna, DNA)

	kernel=WeightedDegreePositionStringKernel(feats_train, feats_train, degree)

	from numpy import zeros,ones,float64,int32
	kernel.set_shifts(10*ones(len(fm_train_dna[0]), dtype=int32))
	kernel.set_position_weights(ones(len(fm_train_dna[0]), dtype=float64))

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('WeightedDegreePositionString')
	kernel_weighted_degree_position_string_modular(*parameter_list[0])
