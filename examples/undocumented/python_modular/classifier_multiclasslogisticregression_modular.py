#!/usr/bin/env python
from tools.multiclass_shared import prepare_data

try:
	from modshogun import MulticlassLogisticRegression
except ImportError:
	print("recompile shogun with Eigen3 support")
	import sys
	sys.exit(0)


[traindat, label_traindat, testdat, label_testdat] = prepare_data(False)

parameter_list = [[traindat,testdat,label_traindat,label_testdat,2.1,1e-5],[traindat,testdat,label_traindat,label_testdat,2.2,1e-5]]

def classifier_multiclasslogisticregression_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,label_test_multiclass=label_testdat,z=1,epsilon=1e-5):
	from modshogun import RealFeatures, MulticlassLabels

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	labels=MulticlassLabels(label_train_multiclass)

	classifier = MulticlassLogisticRegression(z,feats_train,labels)
	classifier.train()

	label_pred = classifier.apply(feats_test)
	out = label_pred.get_labels()

	if label_test_multiclass is not None:
		from modshogun import MulticlassAccuracy
		labels_test = MulticlassLabels(label_test_multiclass)
		evaluator = MulticlassAccuracy()
		acc = evaluator.evaluate(label_pred, labels_test)
		print('Accuracy = %.4f' % acc)
	
	return out

if __name__=='__main__':
	print('MulticlassLogisticRegression')
	classifier_multiclasslogisticregression_modular(*parameter_list[0])
