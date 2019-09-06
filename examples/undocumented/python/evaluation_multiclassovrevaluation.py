#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

parameter_list = [[traindat, label_traindat]]

def evaluation_multiclassovrevaluation(train_fname=traindat, label_fname=label_traindat):
	import shogun as sg
	from shogun import MulticlassLabels

	ground_truth_labels = MulticlassLabels(sg.csv_file(label_fname))
	svm = sg.machine("MulticlassLibLinear", C=1.0,
					labels=ground_truth_labels, seed=1)
	svm.get_global_parallel().set_num_threads(1)
	svm.train(sg.features(sg.csv_file(train_fname)))
	predicted_labels = svm.apply()

	binary_evaluator = sg.evaluation("ROCEvaluation")
	evaluator = sg.evaluation("MulticlassOVREvaluation", binary_evaluation=binary_evaluator)
	mean_roc = evaluator.evaluate(predicted_labels,ground_truth_labels)
	#print mean_roc

	binary_evaluator = sg.evaluation("ContingencyTableEvaluation", type="ACCURACY")
	evaluator = sg.evaluation("MulticlassOVREvaluation", binary_evaluation=binary_evaluator)
	mean_accuracy = evaluator.evaluate(predicted_labels,ground_truth_labels)

	return mean_roc, mean_accuracy, predicted_labels, svm

if __name__=='__main__':
	print('MulticlassOVREvaluation')
	evaluation_multiclassovrevaluation(*parameter_list[0])

