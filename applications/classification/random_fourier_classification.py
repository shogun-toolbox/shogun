#!/usr/bin/env python

#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
# 
#  Written (W) 2013 Evangelos Anagnostopoulos
# 

def parse_arguments():
	import argparse
	parser = argparse.ArgumentParser(description=
		"Solve binary classification problems stored in libsvm format, "
		"using Random Fourier features and SVMOcas")
	parser.add_argument('--dataset', required=True, type=str, 
					help='Path to training dataset in LibSVM format.')
	parser.add_argument('--testset', type=str,
					help='Path to test dataset in LibSVM format.')
	parser.add_argument('-D', default=300, type=int,
					help='The number of samples to use')
	parser.add_argument('-C', default=0.1, type=float,
					help='SVMOcas regularization constant')
	parser.add_argument('--epsilon', default=0.01, type=float,
					help='SVMOcas epsilon parameter')
	parser.add_argument('--width', default=8, type=float,
					help='Width of the Gaussian Kernel to approximate')
	parser.add_argument('--dimension', type=int,
					help='Dimension of input dataset')

	return parser.parse_args()

def evaluate(predicted_labels, labels, prefix="Results"):
	from modshogun import PRCEvaluation, ROCEvaluation, AccuracyMeasure

	prc_evaluator = PRCEvaluation()
	roc_evaluator = ROCEvaluation()
	acc_evaluator = AccuracyMeasure()

	auPRC = prc_evaluator.evaluate(predicted_labels, labels)
	auROC = roc_evaluator.evaluate(predicted_labels, labels)
	acc = acc_evaluator.evaluate(predicted_labels, labels)

	print ('{0}: auPRC = {1:.5f}, auROC = {2:.5f}, acc = {3:.5f} '+
				'({4}% incorrectly classified)').format(
				prefix, auPRC, auROC, acc, (1-acc)*100)

def load_sparse_data(filename, dimension=None):
	input_file = LibSVMFile(args.dataset)
	sparse_feats = SparseRealFeatures()
	label_array = sparse_feats.load_with_labels(input_file)
	labels = BinaryLabels(label_array)

	if dimension!=None:
		sparse_feats.set_num_features(dimension)

	return {'data':sparse_feats, 'labels':labels}

if __name__=='__main__':
	from modshogun import SparseRealFeatures, RandomFourierDotFeatures, GAUSSIAN
	from modshogun import LibSVMFile, BinaryLabels, SVMOcas
	from modshogun import Time
	from numpy import array

	args = parse_arguments()

	print 'Loading training data...'
	sparse_data = load_sparse_data(args.dataset,args.dimension)
	
	kernel_params = array([args.width], dtype=float)
	rf_feats = RandomFourierDotFeatures(sparse_data['data'], args.D, GAUSSIAN,
				kernel_params)

	svm = SVMOcas(args.C, rf_feats, sparse_data['labels'])
	svm.set_epsilon(args.epsilon)
	print 'Starting training.'
	timer = Time()
	svm.train()
	timer.stop()
	print 'Training completed, took {0:.2f}s.'.format(timer.time_diff_sec())

	predicted_labels = svm.apply()
	evaluate(predicted_labels, sparse_data['labels'], 'Training results')

	if args.testset!=None:
		random_coef = rf_feats.get_random_coefficients()
		# removing current dataset from memory in order to load the test dataset,
		# to avoid running out of memory
		rf_feats = None
		svm.set_features(None)
		svm.set_labels(None)
		sparse_data = None

		print 'Loading test data...'
		sparse_data = load_sparse_data(args.testset, args.dimension)
		rf_feats = RandomFourierDotFeatures(sparse_data['data'], args.D, GAUSSIAN,
					kernel_params, random_coef)
		predicted_labels = svm.apply(rf_feats)
		evaluate(predicted_labels, sparse_data['labels'], 'Test results')
