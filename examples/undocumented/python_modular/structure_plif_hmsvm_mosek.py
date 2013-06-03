#!/usr/bin/env python

parameter_list=[[100, 250, 10, 2]]

def structure_plif_hmsvm_mosek (num_examples, example_length, num_features, num_noise_features):
	from shogun.Features   import RealMatrixFeatures
	from shogun.Loss       import HingeLoss
	from shogun.Structure  import TwoStateModel
	from shogun.Evaluation import StructuredAccuracy

	try:
		from shogun.Structure import PrimalMosekSOSVM
	except ImportError:
		print("Mosek not available")
		return

	model = TwoStateModel.simulate_data(num_examples, example_length, num_features, num_noise_features)
	loss = HingeLoss()
	sosvm = PrimalMosekSOSVM(model, loss, model.get_labels())

	sosvm.train()
	#print(sosvm.get_w())

	predicted = sosvm.apply(model.get_features())
	evaluator = StructuredAccuracy()
	acc = evaluator.evaluate(predicted, model.get_labels())
	#print('Accuracy = %.4f' % acc)

if __name__ == '__main__':
	print("PLiF HMSVM Mosek")
	structure_plif_hmsvm_mosek(*parameter_list[0])
