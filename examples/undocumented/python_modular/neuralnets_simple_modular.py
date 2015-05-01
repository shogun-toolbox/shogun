#!/usr/bin/env python
traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_twoclass.dat'

parameter_list = [[traindat,testdat,label_traindat,0.9,1e-3],[traindat,testdat,label_traindat,0.8,1e-2]]

def neuralnets_simple_modular (train_fname, test_fname,
		label_fname, C, epsilon):

	from modshogun import NeuralLayers, NeuralNetwork, RealFeatures, BinaryLabels
	from modshogun import Math_init_random, CSVFile
	Math_init_random(17)

	feats_train=RealFeatures(CSVFile(train_fname))
	feats_test=RealFeatures(CSVFile(test_fname))
	labels=BinaryLabels(CSVFile(label_fname))

	layers = NeuralLayers()
	network = NeuralNetwork(layers.input(feats_train.get_num_features()).linear(50).softmax(2).done())
	network.quick_connect()
	network.initialize()

	network.set_labels(labels)
	network.train(feats_train)
	return network, network.apply_multiclass(feats_test)

if __name__=='__main__':
	print('Neural nets')
	neuralnets_simple_modular(*parameter_list[0])
