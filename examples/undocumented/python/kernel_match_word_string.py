#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()
traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat, 3,1.4,10,3,0,False],[
traindat,testdat, 3,1.4,10,3,0,False]]

def kernel_match_word_string_modular (fm_train_dna=traindat,fm_test_dna=testdat,
degree=3,scale=1.4,size_cache=10,order=3,gap=0,reverse=False):
	from modshogun import MatchWordStringKernel, AvgDiagKernelNormalizer
	from modshogun import StringWordFeatures, StringCharFeatures, DNA

	charfeat=StringCharFeatures(fm_train_dna, DNA)
	feats_train=StringWordFeatures(DNA)
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)

	charfeat=StringCharFeatures(fm_test_dna, DNA)
	feats_test=StringWordFeatures(DNA)
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)

	kernel=MatchWordStringKernel(size_cache, degree)
	kernel.set_normalizer(AvgDiagKernelNormalizer(scale))
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('MatchWordString')
	kernel_match_word_string_modular(*parameter_list[0])

