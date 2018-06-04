#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindna = lm.load_dna('../data/fm_train_dna.dat')
testdna = lm.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindna,testdna,4,0,False,False],[traindna,testdna,3,0,False,False]]

def preprocessor_sortulongstring (fm_train_dna=traindna,fm_test_dna=testdna,order=3,gap=0,reverse=False,use_sign=False):

	from shogun import CommUlongStringKernel
	from shogun import StringCharFeatures, StringUlongFeatures, DNA
	from shogun import SortUlongString


	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_train_dna)
	feats_train=StringUlongFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
	feats_test=StringUlongFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)

	preproc = SortUlongString()
	preproc.fit(feats_train)
        feats_train = preproc.apply(feats_train)
        feats_test = preproc.apply(feats_test)

	kernel=CommUlongStringKernel(feats_train, feats_train, use_sign)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('CommUlongString')
	preprocessor_sortulongstring(*parameter_list[0])
