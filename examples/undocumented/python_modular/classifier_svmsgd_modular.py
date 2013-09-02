#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,0.9,1,6],[traindat,testdat,label_traindat,0.8,1,5]]

def classifier_svmsgd_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_twoclass=label_traindat,C=0.9,num_threads=1,num_iter=5):

	from modshogun import RealFeatures, SparseRealFeatures, BinaryLabels
	from modshogun import SVMSGD

	realfeat=RealFeatures(fm_train_real)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	realfeat=RealFeatures(fm_test_real)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	labels=BinaryLabels(label_train_twoclass)

	svm=SVMSGD(C, feats_train, labels)
	svm.set_epochs(num_iter)
	#svm.io.set_loglevel(0)
	svm.train()

	svm.set_features(feats_test)
	svm.apply().get_labels()
	predictions = svm.apply()
	return predictions, svm, predictions.get_labels()



if __name__=='__main__':
	print('SVMSGD')
	classifier_svmsgd_modular(*parameter_list[0])
