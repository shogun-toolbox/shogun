#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import random
lm=LoadMatrix()

N = 100

random.seed(17)
ground_truth = random.randn(N)
predicted = random.randn(N)

parameter_list = [[ground_truth,predicted]]

def evaluation_meansquarederror (ground_truth, predicted):
	from shogun import RegressionLabels
	import shogun as sg

	ground_truth_labels = RegressionLabels(ground_truth)
	predicted_labels = RegressionLabels(predicted)

	evaluator = sg.create_evaluation("MeanSquaredError")
	mse = evaluator.evaluate(predicted_labels,ground_truth_labels)

	return mse


if __name__=='__main__':
	print('MeanSquaredError')
	evaluation_meansquarederror(*parameter_list[0])

