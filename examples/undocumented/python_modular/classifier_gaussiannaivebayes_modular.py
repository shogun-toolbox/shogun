#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

parameter_list = [[traindat,testdat,label_traindat]]

def classifier_gaussiannaivebayes_modular (train_fname=traindat,test_fname=testdat,label_train_fname=label_traindat):
	from modshogun import RealFeatures, MulticlassLabels, GaussianNaiveBayes, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))
	labels=MulticlassLabels(CSVFile(label_train_fname))

	gnb=GaussianNaiveBayes(feats_train, labels)
	gnb_train = gnb.train()
	output=gnb.apply(feats_test).get_labels()
	return gnb, gnb_train, output

if __name__=='__main__':
	print('GaussianNaiveBayes')
	classifier_gaussiannaivebayes_modular(*parameter_list[0])
