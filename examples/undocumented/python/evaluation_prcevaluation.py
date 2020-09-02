#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import random
lm=LoadMatrix()

ground_truth = lm.load_labels('../data/label_train_twoclass.dat')
random.seed(17)
predicted = random.randn(len(ground_truth))

parameter_list = [[ground_truth,predicted]]

def evaluation_prcevaluation (ground_truth, predicted):
	import shogun as sg
	from shogun import BinaryLabels

	ground_truth_labels = BinaryLabels(ground_truth)
	predicted_labels = BinaryLabels(predicted)

	evaluator = sg.create("PRCEvaluation")
	evaluator.evaluate(predicted_labels,ground_truth_labels)

	return evaluator.get("PRC"), evaluator.get("auPRC")


if __name__=='__main__':
	print('PRCEvaluation')
	evaluation_prcevaluation(*parameter_list[0])

