#!/usr/bin/env python
import shogun as sg
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')
parameter_list=[[traindat,testdat],[traindat,testdat]]

def kernel_local_alignment_string (fm_train_dna=traindat,fm_test_dna=testdat):

	feats_train=sg.create_string_features(fm_train_dna, sg.DNA)
	feats_test=sg.create_string_features(fm_test_dna, sg.DNA)

	kernel=sg.create("LocalAlignmentStringKernel")
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('LocalAlignmentString')
	kernel_local_alignment_string(*parameter_list[0])
