#!/usr/bin/env python
from tools.load import LoadMatrix
import shogun as sg
lm=LoadMatrix()
traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat],[traindat,testdat]]

def kernel_weighted_comm_word_string (fm_train_dna=traindat,fm_test_dna=testdat,order=3,gap=0,reverse=True ):
	from shogun import WeightedCommWordStringKernel
	from shogun import StringWordFeatures, StringCharFeatures, DNA

	charfeat=StringCharFeatures(fm_train_dna, DNA)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc = sg.transformer("SortWordString")
	preproc.fit(feats_train)
	feats_train = preproc.transform(feats_train)

	charfeat=StringCharFeatures(fm_test_dna, DNA)
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test = preproc.transform(feats_test)

	use_sign=False
	kernel = sg.kernel("WeightedCommWordStringKernel", use_sign=use_sign)
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('WeightedCommWordString')
	kernel_weighted_comm_word_string(*parameter_list[0])
