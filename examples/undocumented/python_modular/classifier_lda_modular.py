#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

parameter_list = [[traindat,testdat,label_traindat,3,1],[traindat,testdat,label_traindat,4,1]]

def classifier_lda_modular (train_fname=traindat,test_fname=testdat,label_fname=label_traindat,gamma=3,num_threads=1):
	from modshogun import RealFeatures, BinaryLabels
	from modshogun import LDA, CSVFile

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	labels=BinaryLabels(label_train_twoclass)

	lda=LDA(gamma, feats_train, labels)
	lda.train()

	lda.get_bias()
	lda.get_w()
	lda.set_features(feats_test)
	predictions = lda.apply().get_labels()
	return lda,predictions

if __name__=='__main__':
	print('LDA')
	classifier_lda_modular(*parameter_list[0])
