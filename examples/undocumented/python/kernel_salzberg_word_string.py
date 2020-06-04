#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()
traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')
label_traindat = lm.load_labels('../data/label_train_dna.dat')

parameter_list = [[traindat,testdat,label_traindat,3,0,False],[traindat,testdat,label_traindat,3,0,False]]
def kernel_salzberg_word_string (fm_train_dna=traindat,fm_test_dna=testdat,label_train_dna=label_traindat,
order=3,gap=0,reverse=False):
	from shogun import StringCharFeatures, StringWordFeatures, DNA, BinaryLabels
	from shogun import SalzbergWordStringKernel
	import shogun as sg

	charfeat=StringCharFeatures(fm_train_dna, DNA)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)

	charfeat=StringCharFeatures(fm_test_dna, DNA)
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)

	labels=sg.create_labels(label_train_dna)
	pie=sg.create_machine("PluginEstimate", labels=labels)
	pie.train(feats_train)

	kernel=sg.SalzbergWordStringKernel(feats_train, feats_train, pie, labels)
	km_train=kernel.get_kernel_matrix()

	kernel.init(feats_train, feats_test)
	pie.apply(feats_test).get("labels")
	km_test=kernel.get_kernel_matrix()
	return km_train,km_test,kernel

if __name__=='__main__':
	print('PluginEstimate w/ SalzbergWord')
	kernel_salzberg_word_string(*parameter_list[0])

