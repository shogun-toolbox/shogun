#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')
parameter_list=[[traindat,testdat],[traindat,testdat]]

def kernel_local_alignment_string_modular (fm_train_dna=traindat,fm_test_dna=testdat):

	from modshogun import StringCharFeatures, DNA
	from modshogun import LocalAlignmentStringKernel

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_test=StringCharFeatures(fm_test_dna, DNA)

	kernel=LocalAlignmentStringKernel(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('LocalAlignmentString')
	kernel_local_alignment_string_modular(*parameter_list[0])
