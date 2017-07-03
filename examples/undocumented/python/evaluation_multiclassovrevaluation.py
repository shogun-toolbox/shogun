#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

parameter_list = [[traindat, label_traindat]]

def evaluation_multiclassovrevaluation(train_fname=traindat, label_fname=label_traindat):
	from shogun import MulticlassOVREvaluation,ROCEvaluation
	from shogun import MulticlassLibLinear,RealFeatures,ContingencyTableEvaluation,ACCURACY
	from shogun import MulticlassLabels, Math, CSVFile

	Math.init_random(1)
	ground_truth_labels = MulticlassLabels(CSVFile(label_fname))
	svm = MulticlassLibLinear(1.0,RealFeatures(CSVFile(train_fname)),ground_truth_labels)
	svm.parallel.set_num_threads(1)
	svm.train()
	predicted_labels = svm.apply()

	binary_evaluator = ROCEvaluation()
	evaluator = MulticlassOVREvaluation(binary_evaluator)
	mean_roc = evaluator.evaluate(predicted_labels,ground_truth_labels)
	#print mean_roc

	binary_evaluator = ContingencyTableEvaluation(ACCURACY)
	evaluator = MulticlassOVREvaluation(binary_evaluator)
	mean_accuracy = evaluator.evaluate(predicted_labels,ground_truth_labels)
	#print mean_accuracy
	return mean_roc, mean_accuracy, predicted_labels, svm

if __name__=='__main__':
	print('MulticlassOVREvaluation')
	evaluation_multiclassovrevaluation(*parameter_list[0])

