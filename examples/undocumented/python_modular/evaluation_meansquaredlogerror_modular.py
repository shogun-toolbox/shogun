#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import random
lm=LoadMatrix()

N = 100

random.seed(17)
ground_truth = abs(random.randn(N))
predicted = abs(random.randn(N))

parameter_list = [[ground_truth,predicted]]

def evaluation_meansquaredlogerror_modular (ground_truth, predicted):
	from modshogun import RegressionLabels
	from modshogun import MeanSquaredLogError

	ground_truth_labels = RegressionLabels(ground_truth)
	predicted_labels = RegressionLabels(predicted)

	evaluator = MeanSquaredLogError()
	mse = evaluator.evaluate(predicted_labels,ground_truth_labels)

	return mse


if __name__=='__main__':
	print('EvaluationMeanSquaredLogError')
	evaluation_meansquaredlogerror_modular(*parameter_list[0])

