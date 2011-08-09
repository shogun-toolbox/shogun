# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

ground_truth = LoadMatrix.load_labels('../data/label_train_twoclass.dat')
random.seed(17)
predicted = random.randn(len(ground_truth))

parameter_list = [[ground_truth,predicted]]

def evaluation_rocevaluation_modular(ground_truth, predicted)

	ground_truth_labels = Labels(ground_truth)
	predicted_labels = Labels(predicted)
	
	evaluator = ROCEvaluation()
	evaluator.evaluate(predicted_labels,ground_truth_labels)

	return evaluator.get_ROC(), evaluator.get_auROC()



end
if __FILE__ == $0
	print 'ROCEvaluation'
	evaluation_rocevaluation_modular(*parameter_list[0])


end
