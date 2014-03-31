#!/usr/bin/env python
from tools.multiclass_shared import prepare_data

[traindat, label_traindat, testdat, label_testdat] = prepare_data(False)

parameter_list = [[traindat,testdat,label_traindat,label_testdat,2.1,1,1e-5],[traindat,testdat,label_traindat,label_testdat,2.2,1,1e-5]]

def classifier_multiclassmultipleoutputliblinear_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,label_test_multiclass=label_testdat,width=2.1,C=1,epsilon=1e-5):
	from modshogun import RealFeatures, MulticlassLabels, MultilabelLabels
	from modshogun import MulticlassLibLinear

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	labels=MulticlassLabels(label_train_multiclass)

	classifier = MulticlassLibLinear(C,feats_train,labels)
	classifier.train()

	label_pred = classifier.apply_multilabel_output(feats_test,2)
	out = label_pred.get_labels()
	#print out
	return out

if __name__=='__main__':
	print('MulticlassMultipleOutputLibLinear')
	classifier_multiclassmultipleoutputliblinear_modular(*parameter_list[0])
