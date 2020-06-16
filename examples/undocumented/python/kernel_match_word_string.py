#!/usr/bin/env python
from tools.load import LoadMatrix
import shogun as sg

lm=LoadMatrix()
traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat, 3,1.4,10,3,0,False],[
traindat,testdat, 3,1.4,10,3,0,False]]

def kernel_match_word_string (fm_train_dna=traindat,fm_test_dna=testdat,
degree=3,scale=1.4,size_cache=10,order=3,gap=0,reverse=False):

	charfeat=sg.create_string_features(fm_train_dna, sg.DNA)
	feats_train=sg.create_string_features(charfeat, order-1, order, gap, reverse)

	charfeat=sg.create_string_features(fm_test_dna, sg.DNA)
	feats_test=sg.create_string_features(charfeat, order-1, order, gap, reverse)

	kernel=sg.create_kernel("MatchWordStringKernel", cache_size=size_cache, degree=degree)
	kernel.set_normalizer(sg.create_kernel_normalizer("AvgDiagKernelNormalizer", scale=scale))
	kernel.init(feats_train, feats_train)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('MatchWordString')
	kernel_match_word_string(*parameter_list[0])

