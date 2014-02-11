#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2014 Soumyajit De
#

#!/usr/bin/env python

from tools.load import LoadMatrix

lm=LoadMatrix()
traindat = lm.load_dna('../data/fm_train_dna.dat')
testdat = lm.load_dna('../data/fm_test_dna.dat')
label_traindat = lm.load_labels('../data/label_train_dna.dat')

parameter_list = [[traindat,testdat,label_traindat,1,5,0.9]]

def classifier_ssk_modular (fm_train_dna=traindat,fm_test_dna=testdat,
		label_train_dna=label_traindat,C=1,maxlen=1,decay=1):
	from modshogun import StringCharFeatures, BinaryLabels
	from modshogun import LibSVM, StringSubsequenceKernel, DNA
	from modshogun import ErrorRateMeasure

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_test=StringCharFeatures(fm_test_dna, DNA)
	labels=BinaryLabels(label_train_dna)
	kernel=StringSubsequenceKernel(feats_train, feats_train, maxlen, decay);

	svm=LibSVM(C, kernel, labels);
	svm.train();

	out=svm.apply(feats_train);
	evaluator = ErrorRateMeasure()
	trainerr = evaluator.evaluate(out,labels)
	# print(trainerr)

	kernel.init(feats_train, feats_test)
	predicted_labels=svm.apply(feats_test).get_labels()
	# print predicted_labels

	return predicted_labels

if __name__=='__main__':
	print('SringSubsequenceKernel classification DNA')
	classifier_ssk_modular(*parameter_list[0])
