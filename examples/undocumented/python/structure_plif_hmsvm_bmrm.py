#!/usr/bin/env python

parameter_list=[[50, 125, 10, 2]]

def structure_plif_hmsvm_bmrm (num_examples, example_length, num_features, num_noise_features):
	import shogun as sg
	from shogun import TwoStateModel
	try:
		from shogun import DualLibQPBMSOSVM
	except ImportError:
		print("DualLibQPBMSOSVM not available")
		exit(0)

	model = TwoStateModel.simulate_data(num_examples, example_length, num_features, num_noise_features)
	sosvm = DualLibQPBMSOSVM(model, model.get_labels(), 5000.0)
	sosvm.set_store_train_info(False)

	sosvm.train()
	#print sosvm.get_w()

	predicted = sosvm.apply(model.get_features())
	evaluator = sg.create_evaluation("StructuredAccuracy")
	acc = evaluator.evaluate(predicted, model.get_labels())
	#print('Accuracy = %.4f' % acc)

if __name__ == '__main__':
	print("PLiF HMSVM BMRM")
	structure_plif_hmsvm_bmrm(*parameter_list[0])
