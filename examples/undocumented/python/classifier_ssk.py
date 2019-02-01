#!/usr/bin/env python
# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Soumyajit De

from tools.load import LoadMatrix

lm=LoadMatrix()
traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')
label_traindat = lm.load_labels('../data/label_train_dna.dat')

parameter_list = [[traindat,testdat,label_traindat,1,5,0.9]]

def classifier_ssk (fm_train_dna=traindat,fm_test_dna=testdat,
		label_train_dna=label_traindat,C=1,maxlen=1,decay=1):
	from shogun import StringCharFeatures, BinaryLabels
	from shogun import SubsequenceStringKernel, DNA
	from shogun import ErrorRateMeasure
	import shogun as sg

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_test=StringCharFeatures(fm_test_dna, DNA)
	labels=BinaryLabels(label_train_dna)
	kernel=SubsequenceStringKernel(feats_train, feats_train, maxlen, decay);

	svm=sg.machine("LibSVM", C1=C, C2=C, kernel=kernel, labels=labels);
	svm.train();

	out=svm.apply(feats_train);
	evaluator = ErrorRateMeasure()
	trainerr = evaluator.evaluate(out,labels)
	# print(trainerr)

	kernel.init(feats_train, feats_test)
	predicted_labels=svm.apply(feats_test).get("labels")
	# print predicted_labels

	return predicted_labels

if __name__=='__main__':
	print('SringSubsequenceKernel classification DNA')
	classifier_ssk(*parameter_list[0])
