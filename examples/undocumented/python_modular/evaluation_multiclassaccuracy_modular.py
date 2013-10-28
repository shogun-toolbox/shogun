#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import random
lm=LoadMatrix()

random.seed(17)
ground_truth = lm.load_labels('../data/label_train_multiclass.dat')
predicted = lm.load_labels('../data/label_train_multiclass.dat') * 2

parameter_list = [[ground_truth,predicted]]

def evaluation_multiclassaccuracy_modular (ground_truth, predicted):
	from modshogun import MulticlassLabels
	from modshogun import MulticlassAccuracy

	ground_truth_labels = MulticlassLabels(ground_truth)
	predicted_labels = MulticlassLabels(predicted)

	evaluator = MulticlassAccuracy()
	accuracy = evaluator.evaluate(predicted_labels,ground_truth_labels)

	return accuracy


if __name__=='__main__':
	print('MulticlassAccuracy')
	evaluation_multiclassaccuracy_modular(*parameter_list[0])

