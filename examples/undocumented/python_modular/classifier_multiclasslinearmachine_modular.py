#!/usr/bin/env python
from tools.multiclass_shared import prepare_data

[traindat, label_traindat, testdat, label_testdat] = prepare_data(False)

parameter_list = [[traindat,testdat,label_traindat,label_testdat,2.1,1,1e-5],[traindat,testdat,label_traindat,label_testdat,2.2,1,1e-5]]

def classifier_multiclasslinearmachine_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,label_test_multiclass=label_testdat,width=2.1,C=1,epsilon=1e-5):
	from modshogun import RealFeatures, MulticlassLabels
	from modshogun import LibLinear, L2R_L2LOSS_SVC, LinearMulticlassMachine, MulticlassOneVsOneStrategy, MulticlassOneVsRestStrategy

	feats_train = RealFeatures(fm_train_real)
	feats_test  = RealFeatures(fm_test_real)

	labels = MulticlassLabels(label_train_multiclass)
	
	classifier = LibLinear(L2R_L2LOSS_SVC)
	classifier.set_epsilon(epsilon)
	classifier.set_bias_enabled(True)
	mc_classifier = LinearMulticlassMachine(MulticlassOneVsOneStrategy(), feats_train, classifier, labels)

	mc_classifier.train()
	label_pred = mc_classifier.apply()
	out = label_pred.get_labels()

	if label_test_multiclass is not None:
		from modshogun import MulticlassAccuracy
		labels_test = MulticlassLabels(label_test_multiclass)
		evaluator = MulticlassAccuracy()
		acc = evaluator.evaluate(label_pred, labels_test)
		print('Accuracy = %.4f' % acc)

	return out

if __name__=='__main__':
	print('MulticlassMachine')
	classifier_multiclasslinearmachine_modular(*parameter_list[0])
