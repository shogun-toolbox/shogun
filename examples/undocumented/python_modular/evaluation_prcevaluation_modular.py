from tools.load import LoadMatrix
from numpy import random
lm=LoadMatrix()

ground_truth = lm.load_labels('../data/label_train_twoclass.dat')
random.seed(17)
predicted = random.randn(len(ground_truth))

parameter_list = [[ground_truth,predicted]]

def evaluation_prcevaluation_modular(ground_truth, predicted):
	from shogun.Features import Labels
	from shogun.Evaluation import PRCEvaluation

	ground_truth_labels = Labels(ground_truth)
	predicted_labels = Labels(predicted)
	
	evaluator = PRCEvaluation()
	evaluator.evaluate(predicted_labels,ground_truth_labels)
	
	return evaluator.get_PRC(), evaluator.get_auPRC()


if __name__=='__main__':
	print 'PRCEvaluation'
	evaluation_prcevaluation_modular(*parameter_list[0])

