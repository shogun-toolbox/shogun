# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

N = 100

random.seed(17)
ground_truth = random.randn(N)
predicted = random.randn(N)

parameter_list = [[ground_truth,predicted]]

def evaluation_meansquarederror_modular(ground_truth, predicted)

	ground_truth_labels = Labels(ground_truth)
	predicted_labels = Labels(predicted)
	
	evaluator = MeanSquaredError()
	mse = evaluator.evaluate(predicted_labels,ground_truth_labels)

	return mse



end
if __FILE__ == $0
	puts 'MeanSquaredError'
	evaluation_meansquarederror_modular(*parameter_list[0])


end
