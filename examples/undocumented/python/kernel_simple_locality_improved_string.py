#!/usr/bin/env python
import shogun as sg
from tools.load import LoadMatrix
lm=LoadMatrix()
traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,5,5,1],[traindat,testdat,5,3,2]]

def kernel_simple_locality_improved_string (fm_train_dna=traindat,fm_test_dna=testdat,
	length=5,inner_degree=5,outer_degree=1):

	feats_train=sg.create_string_features(fm_train_dna, sg.DNA)
	feats_test=sg.create_string_features(fm_test_dna, sg.DNA)

	kernel=sg.create("SimpleLocalityImprovedStringKernel", length=length, inner_degree=inner_degree, outer_degree=outer_degree)
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('SimpleLocalityImprovedString')
	kernel_simple_locality_improved_string(*parameter_list[0])
