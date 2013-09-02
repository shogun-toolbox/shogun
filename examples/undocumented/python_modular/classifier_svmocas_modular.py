#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')

parameter_list = [[traindat,testdat,label_traindat,0.9,1e-5,1],[traindat,testdat,label_traindat,0.8,1e-5,1]]

def classifier_svmocas_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_twoclass=label_traindat,C=0.9,epsilon=1e-5,num_threads=1):

	from modshogun import RealFeatures, SparseRealFeatures, BinaryLabels
	from modshogun import SVMOcas

	realfeat=RealFeatures(fm_train_real)
	feats_train=SparseRealFeatures()
	feats_train.obtain_from_simple(realfeat)
	realfeat=RealFeatures(fm_test_real)
	feats_test=SparseRealFeatures()
	feats_test.obtain_from_simple(realfeat)

	labels=BinaryLabels(label_train_twoclass)

	svm=SVMOcas(C, feats_train, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.set_bias_enabled(False)
	svm.train()

	svm.set_features(feats_test)
	svm.apply().get_labels()
	predictions = svm.apply()
	return predictions, svm, predictions.get_labels()

if __name__=='__main__':
	print('SVMOcas')
	classifier_svmocas_modular(*parameter_list[0])
