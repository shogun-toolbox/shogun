#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_twoclass.dat'

parameter_list = [[traindat,testdat,label_traindat,0.9,1,6],[traindat,testdat,label_traindat,0.8,1,5]]

def classifier_svmsgd_modular (train_fname=traindat,test_fname=testdat,label_fname=label_traindat,C=0.9,num_threads=1,num_iter=5):
	from modshogun import RealFeatures, SparseRealFeatures, BinaryLabels
	from modshogun import SVMSGD, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))
	labels=BinaryLabels(CSVFile(label_fname))

	svm=SVMSGD(C, feats_train, labels)
	svm.set_epochs(num_iter)
	#svm.io.set_loglevel(0)
	svm.train()

	bias=svm.get_bias()
	w=w.get_w()
	predictions = svm.apply(feats_test)
	return predictions, svm, predictions.get_labels()

if __name__=='__main__':
	print('SVMSGD')
	classifier_svmsgd_modular(*parameter_list[0])
