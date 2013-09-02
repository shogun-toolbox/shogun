#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import random
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat  = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_multiclass.dat')

parameter_list = [[traindat, label_traindat]]

def evaluation_multiclassovrevaluation_modular (traindat, label_traindat):
	from modshogun import MulticlassLabels
	from modshogun import MulticlassOVREvaluation,ROCEvaluation
	from modshogun import MulticlassLibLinear,RealFeatures,ContingencyTableEvaluation,ACCURACY
	from modshogun import Math
	
	Math.init_random(1)

	ground_truth_labels = MulticlassLabels(label_traindat)
	svm = MulticlassLibLinear(1.0,RealFeatures(traindat),MulticlassLabels(label_traindat))
	svm.parallel.set_num_threads(1)
	svm.train()
	predicted_labels = svm.apply()
	
	binary_evaluator = ROCEvaluation()
	evaluator = MulticlassOVREvaluation(binary_evaluator)
	mean_roc = evaluator.evaluate(predicted_labels,ground_truth_labels)
	#print mean_roc
	
	binary_evaluator = ContingencyTableEvaluation(ACCURACY)
	evaluator = MulticlassOVREvaluation(binary_evaluator)
	mean_accuracy = evaluator.evaluate(predicted_labels,ground_truth_labels)
	#print mean_accuracy

	return mean_roc, mean_accuracy, predicted_labels, svm


if __name__=='__main__':
	print('MulticlassOVREvaluation')
	evaluation_multiclassovrevaluation_modular(*parameter_list[0])

