from tools.load import LoadMatrix
from numpy import random
lm=LoadMatrix()

ground_truth = lm.load_labels('../data/label_train_twoclass.dat')
random.seed(17)
predicted = random.randn(len(ground_truth))

parameter_list = [[ground_truth,predicted]]

def evaluation_rocevaluation_modular(ground_truth, predicted):
	from shogun.Features import Labels
	from shogun.Evaluation import ROCEvaluation

	ground_truth_labels = Labels(ground_truth)
	predicted_labels = Labels(predicted)
	
	evaluator = ROCEvaluation()
	evaluator.evaluate(predicted_labels,ground_truth_labels)

	return evaluator.get_ROC(), evaluator.get_auROC()


if __name__=='__main__':
	print 'ROCEvaluation'
	evaluation_rocevaluation_modular(*parameter_list[0])

