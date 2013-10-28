require 'modshogun'
require 'load'

ground_truth = load_labels('../data/label_train_twoclass.dat')
math.randomseed(17)

predicted = {}
for i = 1, #ground_truth do
	table.insert(predicted, math.random())
end
parameter_list = {{ground_truth,predicted}}

function evaluation_contingencytableevaluation_modular(ground_truth, predicted)

	ground_truth_labels = modshogun.BinaryLabels(ground_truth)
	predicted_labels = modshogun.BinaryLabels(predicted)

	base_evaluator = modshogun.ContingencyTableEvaluation()
	base_evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = modshogun.AccuracyMeasure()
	accuracy = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = modshogun.ErrorRateMeasure()
	errorrate = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = modshogun.BALMeasure()
	bal = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = modshogun.WRACCMeasure()
	wracc = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = modshogun.F1Measure()
	f1 = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = modshogun.CrossCorrelationMeasure()
	crosscorrelation = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = modshogun.RecallMeasure()
	recall = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = modshogun.PrecisionMeasure()
	precision = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = modshogun.SpecificityMeasure()
	specificity = evaluator:evaluate(predicted_labels,ground_truth_labels)

	return accuracy, errorrate, bal, wracc, f1, crosscorrelation, recall, precision, specificity
end

if debug.getinfo(3) == nill then
	print 'ContingencyTableEvaluation'
	evaluation_contingencytableevaluation_modular(unpack(parameter_list[1]))
end

