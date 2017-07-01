#!/usr/bin/env python
from tools.multiclass_shared import prepare_data

[traindat, label_traindat, testdat, label_testdat] = prepare_data()

parameter_list = [[traindat,testdat,label_traindat,2.1,1,1e-5],[traindat,testdat,label_traindat,2.2,1,1e-5]]

def classifier_multiclassmachine_modular (fm_train_real=traindat,fm_test_real=testdat,label_train_multiclass=label_traindat,width=2.1,C=1,epsilon=1e-5):
	from modshogun import RealFeatures, MulticlassLabels
	from modshogun import GaussianKernel
	from modshogun import LibSVM, KernelMulticlassMachine, MulticlassOneVsRestStrategy

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)
	kernel=GaussianKernel(feats_train, feats_train, width)

	labels=MulticlassLabels(label_train_multiclass)

	classifier = LibSVM()
	classifier.set_epsilon(epsilon)
	#print labels.get_labels()
	mc_classifier = KernelMulticlassMachine(MulticlassOneVsRestStrategy(),kernel,classifier,labels)
	mc_classifier.train()

	kernel.init(feats_train, feats_test)
	out = mc_classifier.apply().get_labels()
	return out

if __name__=='__main__':
	print('MulticlassMachine')
	classifier_multiclassmachine_modular(*parameter_list[0])
