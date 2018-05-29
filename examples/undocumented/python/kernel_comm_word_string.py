#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')
parameter_list = [[traindat,testdat,4,0,False, False],[traindat,testdat,4,0,False,False]]

def kernel_comm_word_string (fm_train_dna=traindat, fm_test_dna=testdat, order=3, gap=0, reverse = False, use_sign = False):

	from shogun import CommWordStringKernel
	from shogun import StringWordFeatures, StringCharFeatures, DNA
	from shogun import SortWordString

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_train_dna)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc = SortWordString()
	preproc.fit(feats_train)
	feats_train = preproc.transform(feats_train)

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test = preproc.transform(feats_test)

	kernel=CommWordStringKernel(feats_train, feats_train, use_sign)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('CommWordString')
	kernel_comm_word_string(*parameter_list[0])
