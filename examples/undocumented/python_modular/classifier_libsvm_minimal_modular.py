#!/usr/bin/env python
from numpy import mean, sign

from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,2.1,1]]

def classifier_libsvm_minimal_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_twoclass=label_traindat,width=2.1,C=1):
	from modshogun import RealFeatures, BinaryLabels
	from modshogun import LibSVM
	from modshogun import GaussianKernel

	feats_train=RealFeatures(fm_train_real);
	feats_test=RealFeatures(fm_test_real);
	kernel=GaussianKernel(feats_train, feats_train, width);

	labels=BinaryLabels(label_train_twoclass);
	svm=LibSVM(C, kernel, labels);
	svm.train();

	kernel.init(feats_train, feats_test);
	out=svm.apply().get_labels();
	testerr=mean(sign(out)!=label_train_twoclass)
	#print(testerr)

if __name__=='__main__':
	print('LibSVM Minimal')
	classifier_libsvm_minimal_modular(*parameter_list[0])
