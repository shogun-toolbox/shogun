#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

parameter_list = [[traindat,testdat,label_traindat]]

def classifier_conjugateindex_modular (train_fname=traindat,test_fname=testdat,label_fname=label_traindat):
	from modshogun import RealFeatures, MulticlassLabels, ConjugateIndex, CSVFile

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))

	labels = MulticlassLabels(CSVFile(label_fname))

	ci = ConjugateIndex(feats_train, labels)
	ci.train()

	res = ci.apply(feats_test).get_labels()
	return ci, res

if __name__=='__main__':
	print('ConjugateIndex')
	classifier_conjugateindex_modular(*parameter_list[0])
