#!/usr/bin/env python

parameter_list=[[100, 250, 10, 2]]

def structure_plif_hmsvm_mosek (num_examples, example_length, num_features, num_noise_features):
	import shogun as sg
	from shogun import TwoStateModel

	try:
		from shogun import PrimalMosekSOSVM
	except ImportError:
		print("Mosek not available")
		return

	model = TwoStateModel.simulate_data(num_examples, example_length, num_features, num_noise_features)
	sosvm = PrimalMosekSOSVM(model, model.get_labels())

	sosvm.train()
	#print(sosvm.get_w())

	predicted = sosvm.apply(model.get_features())
	evaluator = sg.create_evaluation("StructuredAccuracy")
	acc = evaluator.evaluate(predicted, model.get_labels())
	#print('Accuracy = %.4f' % acc)

if __name__ == '__main__':
	print("PLiF HMSVM Mosek")
	structure_plif_hmsvm_mosek(*parameter_list[0])
