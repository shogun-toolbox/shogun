#!/usr/bin/env python
from tools.multiclass_shared import prepare_data

[traindat, label_traindat, testdat, label_testdat] = prepare_data(True)

parameter_list = [[traindat,testdat,label_traindat,label_testdat,2.1,1,1e-5],[traindat,testdat,label_traindat,label_testdat,2.2,1,1e-5]]

def classifier_multiclass_relaxedtree (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,label_test_multiclass=label_testdat,lawidth=2.1,C=1,epsilon=1e-5):
	from modshogun import RealFeatures, MulticlassLabels
	from modshogun import RelaxedTree, MulticlassLibLinear
	from modshogun import GaussianKernel

	#print('Working on a problem of %d features and %d samples' % fm_train_real.shape)

	feats_train = RealFeatures(fm_train_real)

	labels = MulticlassLabels(label_train_multiclass)

	machine = RelaxedTree()
	machine.set_machine_for_confusion_matrix(MulticlassLibLinear())
	machine.set_kernel(GaussianKernel())
	machine.set_labels(labels)
	machine.train(feats_train)

	label_pred = machine.apply_multiclass(RealFeatures(fm_test_real))
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
	classifier_multiclass_relaxedtree(*parameter_list[0])

