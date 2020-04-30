#!/usr/bin/env python
from tools.load import LoadMatrix
import shogun as sg
lm=LoadMatrix()
traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')

parameter_list=[[traindat,testdat],[traindat,testdat]]

def kernel_linear_string (fm_train_dna=traindat,fm_test_dna=testdat):
	from shogun import StringCharFeatures, DNA
	from shogun import LinearStringKernel

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_test=StringCharFeatures(fm_test_dna, DNA)

	kernel=sg.create_kernel("LinearStringKernel")
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	from tools.load import LoadMatrix
	print('LinearString')
	kernel_linear_string(*parameter_list[0])
