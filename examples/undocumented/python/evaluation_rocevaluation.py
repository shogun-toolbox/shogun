#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import random
lm=LoadMatrix()

ground_truth = lm.load_labels('../data/label_train_twoclass.dat')
random.seed(17)
predicted = random.randn(len(ground_truth))

parameter_list = [[ground_truth,predicted]]

def evaluation_rocevaluation (ground_truth, predicted):
	import shogun as sg
	from shogun import BinaryLabels

	ground_truth_labels = BinaryLabels(ground_truth)
	predicted_labels = BinaryLabels(predicted)

	evaluator = sg.create("ROCEvaluation")
	evaluator.evaluate(predicted_labels,ground_truth_labels)

	return evaluator.get("ROC"), evaluator.get("auROC")

if __name__=='__main__':
	print('ROCEvaluation')
	evaluation_rocevaluation(*parameter_list[0])

