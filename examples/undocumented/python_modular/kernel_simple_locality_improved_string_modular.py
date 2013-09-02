#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()
traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,5,5,1],[traindat,testdat,5,3,2]]

def kernel_simple_locality_improved_string_modular (fm_train_dna=traindat,fm_test_dna=testdat,
	length=5,inner_degree=5,outer_degree=1 ):
	
	from modshogun import StringCharFeatures, DNA
	from modshogun import SimpleLocalityImprovedStringKernel, MSG_DEBUG

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	#feats_train.io.set_loglevel(MSG_DEBUG)
	feats_test=StringCharFeatures(fm_test_dna, DNA)


	kernel=SimpleLocalityImprovedStringKernel(
		feats_train, feats_train, length, inner_degree, outer_degree)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('SimpleLocalityImprovedString')
	kernel_simple_locality_improved_string_modular(*parameter_list[0])
