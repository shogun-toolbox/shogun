#!/usr/bin/env python

parameter_list=[[100, 250, 10, 2]]

def structure_plif_hmsvm_bmrm (num_examples, example_length, num_features, num_noise_features):
	from modshogun import RealMatrixFeatures, TwoStateModel, DualLibQPBMSOSVM, StructuredAccuracy

	model = TwoStateModel.simulate_data(num_examples, example_length, num_features, num_noise_features)
	sosvm = DualLibQPBMSOSVM(model, model.get_labels(), 5000.0)

	sosvm.train()
	#print sosvm.get_w()

	predicted = sosvm.apply(model.get_features())
	evaluator = StructuredAccuracy()
	acc = evaluator.evaluate(predicted, model.get_labels())
	#print('Accuracy = %.4f' % acc)

if __name__ == '__main__':
	print("PLiF HMSVM BMRM")
	structure_plif_hmsvm_bmrm(*parameter_list[0])
