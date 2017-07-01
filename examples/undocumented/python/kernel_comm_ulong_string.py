#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat =lm.load_dna('../data/fm_train_dna.dat')
testdat =  lm.load_dna('../data/fm_test_dna.dat')
parameter_list = [[traindat,testdat,3,0,False ],[traindat,testdat,4,0,False]]

def kernel_comm_ulong_string_modular (fm_train_dna=traindat,fm_test_dna=testdat, order=3, gap=0, reverse = False):

	from modshogun import CommUlongStringKernel
	from modshogun import StringUlongFeatures, StringCharFeatures, DNA
	from modshogun import SortUlongString

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_train_dna)
	feats_train=StringUlongFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortUlongString()
	preproc.init(feats_train)
	feats_train.add_preprocessor(preproc)
	feats_train.apply_preprocessor()


	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
	feats_test=StringUlongFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test.add_preprocessor(preproc)
	feats_test.apply_preprocessor()

	use_sign=False

	kernel=CommUlongStringKernel(feats_train, feats_train, use_sign)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('CommUlongString')
	kernel_comm_ulong_string_modular(*parameter_list[0])
