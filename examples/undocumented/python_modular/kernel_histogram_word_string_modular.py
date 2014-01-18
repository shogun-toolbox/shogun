#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')
label_traindat = lm.load_labels('../data/label_train_dna.dat')
parameter_list=[[traindat,testdat,label_traindat,3,0,False],[traindat,testdat,label_traindat,3,0,False]]

def kernel_histogram_word_string_modular (fm_train_dna=traindat,fm_test_dna=testdat,label_train_dna=label_traindat,order=3,gap=0,reverse=False):

	from modshogun import StringCharFeatures, StringWordFeatures, DNA, BinaryLabels
	from modshogun import HistogramWordStringKernel, AvgDiagKernelNormalizer
	from modshogun import PluginEstimate#, MSG_DEBUG

	reverse = reverse
	charfeat=StringCharFeatures(DNA)
	#charfeat.io.set_loglevel(MSG_DEBUG)
	charfeat.set_features(fm_train_dna)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)

	pie=PluginEstimate()
	labels=BinaryLabels(label_train_dna)
	pie.set_labels(labels)
	pie.set_features(feats_train)
	pie.train()

	kernel=HistogramWordStringKernel(feats_train, feats_train, pie)
	kernel.set_normalizer(AvgDiagKernelNormalizer())
	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	pie.set_features(feats_test)
	pie.apply().get_labels()
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('PluginEstimate w/ HistogramWord')
	kernel_histogram_word_string_modular(*parameter_list[0])
