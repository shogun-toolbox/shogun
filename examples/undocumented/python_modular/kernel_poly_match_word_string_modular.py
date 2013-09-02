#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()
traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,2,True,3,0,False],[traindat,testdat,2,True,3,0,False]]

def kernel_poly_match_word_string_modular (fm_train_dna=traindat,fm_test_dna=testdat,
degree=2,inhomogene=True,order=3,gap=0,reverse=False):
	from modshogun import PolyMatchWordStringKernel
	from modshogun import StringWordFeatures, StringCharFeatures, DNA



	charfeat=StringCharFeatures(fm_train_dna, DNA)
	feats_train=StringWordFeatures(DNA)
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)

	charfeat=StringCharFeatures(fm_test_dna, DNA)
	feats_test=StringWordFeatures(DNA)
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)

	kernel=PolyMatchWordStringKernel(feats_train, feats_train, degree, inhomogene)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('PolyMatchWordString')
	kernel_poly_match_word_string_modular(*parameter_list[0])
