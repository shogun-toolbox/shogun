# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

random.seed(17)
ground_truth = LoadMatrix.load_labels('../data/label_train_multiclass.dat')
predicted = LoadMatrix.load_labels('../data/label_train_multiclass.dat') * 2

parameter_list = [[ground_truth,predicted]]

def evaluation_multiclassaccuracy_modular(ground_truth, predicted)

	ground_truth_labels = Labels(ground_truth)
	predicted_labels = Labels(predicted)
	
	evaluator = MulticlassAccuracy()
	accuracy = evaluator.evaluate(predicted_labels,ground_truth_labels)
	
	return accuracy



end
if __FILE__ == $0
	print 'MulticlassAccuracy'
	evaluation_multiclassaccuracy_modular(*parameter_list[0])


end
