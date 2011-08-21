# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

ground_truth = LoadMatrix.load_labels('../data/label_train_twoclass.dat')
random.seed(17)
predicted = random.randn(len(ground_truth))

parameter_list = [[ground_truth,predicted]]

def evaluation_prcevaluation_modular(ground_truth, predicted)

	ground_truth_labels = Labels(ground_truth)
	predicted_labels = Labels(predicted)
	
	evaluator = PRCEvaluation()
	evaluator.evaluate(predicted_labels,ground_truth_labels)
	
	return evaluator.get_PRC(), evaluator.get_auPRC()



end
if __FILE__ == $0
	puts 'PRCEvaluation'
	evaluation_prcevaluation_modular(*parameter_list[0])


end
