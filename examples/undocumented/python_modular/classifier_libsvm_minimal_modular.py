#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_twoclass.dat'

parameter_list = [[traindat,testdat,label_traindat,2.1,1]]

def classifier_libsvm_minimal_modular (train_fname=traindat,test_fname=testdat,label_fname=label_traindat,width=2.1,C=1):
	from modshogun import RealFeatures, BinaryLabels
	from modshogun import LibSVM, GaussianKernel, CSVFile
	from modshogun import ErrorRateMeasure

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))
	labels=BinaryLabels(CSVFile(label_fname))
	kernel=GaussianKernel(feats_train, feats_train, width);

	svm=LibSVM(C, kernel, labels);
	svm.train();

	out=svm.apply(feats_train);
	evaluator = ErrorRateMeasure()
	testerr = evaluator.evaluate(out,labels)
	#print(testerr)

if __name__=='__main__':
	print('LibSVM Minimal')
	classifier_libsvm_minimal_modular(*parameter_list[0])
