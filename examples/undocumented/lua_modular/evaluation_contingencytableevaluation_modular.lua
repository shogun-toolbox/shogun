require 'shogun'
require 'load'

ground__truth = load_labels('../data/label_train_twoclass.dat')
math.randomseed(17)

--predicted = random.randn(len(ground_truth))
predicte = {}
for i = 1, #ground__truth do
	table.insert(predicte, math.random())
end
parameter_list = {{ground__truth,predicte}}

function evaluation_contingencytableevaluation_modular(ground_truth, predicted)

	ground_truth_labels = Labels(ground_truth)
	predicted_labels = Labels(predicted)
	
	base_evaluator = ContingencyTableEvaluation()
	base_evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = AccuracyMeasure()	
	accuracy = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = ErrorRateMeasure()
	errorrate = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = BALMeasure()
	bal = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = WRACCMeasure()
	wracc = evaluator:evaluate(predicted_labels,ground_truth_labels)
 
	evaluator = F1Measure()
	f1 = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = CrossCorrelationMeasure()
	crosscorrelation = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = RecallMeasure()
	recall = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = PrecisionMeasure()
	precision = evaluator:evaluate(predicted_labels,ground_truth_labels)

	evaluator = SpecificityMeasure()
	specificity = evaluator:evaluate(predicted_labels,ground_truth_labels)

	return accuracy, errorrate, bal, wracc, f1, crosscorrelation, recall, precision, specificity
end

print 'ContingencyTableEvaluation'
evaluation_contingencytableevaluation_modular(unpack(parameter_list[1]))

