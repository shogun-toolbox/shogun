#!/usr/bin/env python
from tools.load import LoadMatrix
from numpy import random
lm=LoadMatrix()

N = 100

random.seed(17)
ground_truth = random.randn(N)
predicted = random.randn(N)

parameter_list = [[ground_truth,predicted]]

def evaluation_meansquarederror_modular (ground_truth, predicted):
	from shogun.Features import RegressionLabels
	from shogun.Evaluation import MeanSquaredError

	ground_truth_labels = RegressionLabels(ground_truth)
	predicted_labels = RegressionLabels(predicted)
	
	evaluator = MeanSquaredError()
	mse = evaluator.evaluate(predicted_labels,ground_truth_labels)

	return mse


if __name__=='__main__':
	print('MeanSquaredError')
	evaluation_meansquarederror_modular(*parameter_list[0])

