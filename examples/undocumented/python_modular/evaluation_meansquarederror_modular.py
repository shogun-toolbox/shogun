from tools.load import LoadMatrix
from numpy import random
lm=LoadMatrix()

N = 100

NArray.srand(17)
ground_truth = NArray.float(N).randomn
predicted = NArray.float(N).randomn

parameter_list = [[ground_truth,predicted]]

def evaluation_meansquarederror_modular(ground_truth, predicted):
	from shogun.Features import Labels
	from shogun.Evaluation import MeanSquaredError

	ground_truth_labels = Modshogun::Labels.new(ground_truth)
	predicted_labels = Modshogun::Labels.new(predicted)
	
	evaluator = Modshogun::MeanSquaredError.new
	mse = evaluator.evaluate(predicted_labels,ground_truth_labels)

	return mse


if __name__=='__main__':
	print 'MeanSquaredError'
	evaluation_meansquarederror_modular(*parameter_list[0])

