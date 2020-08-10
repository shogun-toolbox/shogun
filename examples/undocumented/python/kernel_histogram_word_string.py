#!/usr/bin/env python
from tools.load import LoadMatrix
import shogun as sg
lm=LoadMatrix()

traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')
label_traindat = lm.load_labels('../data/label_train_dna.dat')
parameter_list=[[traindat,testdat,label_traindat,1,1e1, 1e0],[traindat,testdat,label_traindat,1,1e4,1e4]]

def kernel_histogram_word_string (fm_train_dna=traindat,fm_test_dna=testdat,label_train_dna=label_traindat,order=3,ppseudo_count=1,npseudo_count=1):

	charfeat=sg.create_string_features(fm_train_dna, sg.DNA)
	feats_train=sg.create_string_features(charfeat, order-1, order, 0, False)

	charfeat=sg.create_string_features(fm_test_dna, sg.DNA)
	feats_test=sg.create_string_features(charfeat, order-1, order, 0, False)

	labels=sg.create_labels(label_train_dna)
	pie=sg.create_machine("PluginEstimate", pos_pseudo=ppseudo_count, neg_pseudo=npseudo_count)
	pie.train(feats_train, labels)

	kernel=sg.create_kernel("HistogramWordStringKernel", estimate=pie)
	kernel.init(feats_train, feats_train)
	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	pie.apply(feats_test).get("labels")
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('PluginEstimate w/ HistogramWord')
	kernel_histogram_word_string(*parameter_list[0])
