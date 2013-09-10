#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_twoclass.dat'

parameter_list = [[traindat,testdat,label_traindat,3,1],[traindat,testdat,label_traindat,4,1]]

def classifier_lda_modular (train_fname=traindat,test_fname=testdat,label_fname=label_traindat,gamma=3,num_threads=1):
	from modshogun import RealFeatures, BinaryLabels, LDA, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))
	labels=BinaryLabels(CSVFile(label_fname))

	lda=LDA(gamma, feats_train, labels)
	lda.train()

	bias=lda.get_bias()
	w=lda.get_w()
	predictions = lda.apply(feats_test).get_labels()
	return lda,predictions

if __name__=='__main__':
	print('LDA')
	classifier_lda_modular(*parameter_list[0])
