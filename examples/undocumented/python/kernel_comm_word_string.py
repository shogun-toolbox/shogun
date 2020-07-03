#!/usr/bin/env python
from tools.load import LoadMatrix
import shogun as sg
lm=LoadMatrix()

traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')
parameter_list = [[traindat,testdat,4,0,False, False],[traindat,testdat,4,0,False,False]]

def kernel_comm_word_string (fm_train_dna=traindat, fm_test_dna=testdat, order=3, gap=0, reverse = False, use_sign = False):

	charfeat=sg.create_string_features(fm_train_dna, sg.DNA)
	feats_train=sg.create_string_features(charfeat, order-1, order, gap, reverse)
	preproc = sg.create_transformer("SortWordString")
	preproc.fit(feats_train)
	feats_train = preproc.transform(feats_train)

	charfeat=sg.create_string_features(fm_test_dna, sg.DNA)
	feats_test=sg.create_string_features(charfeat, order-1, order, gap, reverse)
	feats_test = preproc.transform(feats_test)

	kernel=sg.create_kernel("CommWordStringKernel", use_sign=use_sign)
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('CommWordString')
	kernel_comm_word_string(*parameter_list[0])
