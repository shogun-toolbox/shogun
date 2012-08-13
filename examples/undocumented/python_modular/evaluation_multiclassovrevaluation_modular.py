from tools.load import LoadMatrix
from numpy import random
lm=LoadMatrix()

random.seed(17)
ground_truth = lm.load_labels('../data/label_train_multiclass.dat')

parameter_list = [[ground_truth]]

def evaluation_multiclassovrevaluation_modular(ground_truth):
	from shogun.Features import MulticlassLabels
	from shogun.Evaluation import MulticlassAccuracy,ROCEvaluation

	ground_truth_labels = MulticlassLabels(ground_truth)
	predicted_labels = MulticlassLabels(ground_truth)
	
	binary_evaluator = ROCEvaluation()
	evaluator = MulticlassAccuracy(binary_evaluator)
	mean_roc = evaluator.evaluate(predicted_labels,ground_truth_labels)
	print mean_roc

	return mean_roc


if __name__=='__main__':
	print('MulticlassOVREvaluation')
	evaluation_multiclassovrevaluation_modular(*parameter_list[0])

