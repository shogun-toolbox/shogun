#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')
label_traindat = lm.load_labels('../data/label_train_dna.dat')
parameter_list=[[traindat,testdat,label_traindat,1,1e1, 1e0],[traindat,testdat,label_traindat,1,1e4,1e4]]

def kernel_histogram_word_string (fm_train_dna=traindat,fm_test_dna=testdat,label_train_dna=label_traindat,order=3,ppseudo_count=1,npseudo_count=1):

	from shogun import StringCharFeatures, StringWordFeatures, DNA
	import shogun as sg

	charfeat=StringCharFeatures(DNA)
	#charfeat.io.set_loglevel(MSG_DEBUG)
	charfeat.set_features(fm_train_dna)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, 0, False)

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, 0, False)

	labels=sg.create_labels(label_train_dna)
	pie=sg.create_machine("PluginEstimate", pos_pseudo=ppseudo_count, neg_pseudo=npseudo_count, labels=labels)
	pie.train(feats_train)

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
