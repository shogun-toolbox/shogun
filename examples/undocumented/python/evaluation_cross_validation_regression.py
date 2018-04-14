#!/usr/bin/env python
#
# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Heiko Strathmann

traindat = '../data/fm_train_real.dat'
label_traindat = '../data/label_train_twoclass.dat'

parameter_list = [[traindat,label_traindat,0.8,1e-6],[traindat,label_traindat,0.9,1e-7]]

def evaluation_cross_validation_regression (train_fname=traindat,label_fname=label_traindat,width=0.8,tau=1e-6):
	from shogun import CrossValidation, CrossValidationResult
	from shogun import MeanSquaredError, CrossValidationSplitting
	from shogun import RegressionLabels, RealFeatures
	from shogun import GaussianKernel, KernelRidgeRegression, CSVFile

	# training data
	features=RealFeatures(CSVFile(train_fname))
	labels=RegressionLabels(CSVFile(label_fname))

	# kernel and predictor
	kernel=GaussianKernel()
	predictor=KernelRidgeRegression(tau, kernel, labels)

	# splitting strategy for 5 fold cross-validation (for classification its better
	# to use "StratifiedCrossValidation", but here, the std x-val is used
	splitting_strategy=CrossValidationSplitting(labels, 5)

	# evaluation method
	evaluation_criterium=MeanSquaredError()

	# cross-validation instance
	cross_validation=CrossValidation(predictor, features, labels,
			splitting_strategy, evaluation_criterium)

	# (optional) repeat x-val 10 times
	cross_validation.set_num_runs(10)

	# (optional) tell machine to precompute kernel matrix. speeds up. may not work
	predictor.data_lock(labels, features)

	# perform cross-validation and print(results)
	result=cross_validation.evaluate()
	#print("mean:", result.mean)

if __name__=='__main__':
	print('Evaluation CrossValidationClassification')
	evaluation_cross_validation_regression(*parameter_list[0])
