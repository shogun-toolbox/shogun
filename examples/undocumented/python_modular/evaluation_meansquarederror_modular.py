from tools.load import LoadMatrix
from numpy import random
lm=LoadMatrix()

N = 100

numpy.random.seed(17)
ground_truth = random.randn(N)
predicted = random.randn(N)

parameter_list = [[ground_truth,predicted]]

def evaluation_meansquarederror_modular(ground_truth, predicted):
	from shogun.Features import Labels
	from shogun.Evaluation import MeanSquaredError

	ground_truth_labels = Labels(ground_truth)
	predicted_labels = Labels(predicted)
	
	evaluator = MeanSquaredError()
	mse = evaluator.evaluate(predicted_labels,ground_truth_labels)

	return mse


if __name__=='__main__':
	print 'MeanSquaredError'
	evaluation_meansquarederror_modular(*parameter_list[0])

