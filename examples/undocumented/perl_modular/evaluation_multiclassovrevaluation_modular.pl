#!/usr/bin/env python
#!/usr/bin/env perl
from tools.load import LoadMatrix
from numpy import random
lm=LoadMatrix()

random.seed(17)
from tools.multiclass_shared import prepare_data
[traindat, label_traindat, testdat, label_testdat] = prepare_data(False)

parameter_list = [[traindat, label_traindat, testdat, label_testdat]]

def evaluation_multiclassovrevaluation_modular (traindat, label_traindat, testdat, label_testdat):
	from shogun.Features import MulticlassLabels
	from shogun.Evaluation import MulticlassOVREvaluation,ROCEvaluation
	from modshogun import MulticlassLibLinear,RealFeatures,ContingencyTableEvaluation,ACCURACY

	ground_truth_labels = MulticlassLabels(label_traindat)
	svm = MulticlassLibLinear(1.0,RealFeatures(traindat),MulticlassLabels(label_traindat))
	svm.train()
	predicted_labels = svm.apply()
	
	binary_evaluator = ROCEvaluation()
	evaluator = MulticlassOVREvaluation(binary_evaluator)
	mean_roc = evaluator.evaluate(predicted_labels,ground_truth_labels)
	print mean_roc
	
	binary_evaluator = ContingencyTableEvaluation(ACCURACY)
	evaluator = MulticlassOVREvaluation(binary_evaluator)
	mean_accuracy = evaluator.evaluate(predicted_labels,ground_truth_labels)
	print mean_accuracy

	return mean_roc, mean_accuracy


if __name__=='__main__':
	print('MulticlassOVREvaluation')
	evaluation_multiclassovrevaluation_modular(*parameter_list[0])

