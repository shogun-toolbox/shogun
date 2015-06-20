#!/usr/bin/env python
from numpy import array, random

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

# set input attribute as not nominal (ie. continuous)
feattypes = array([False])

parameter_list = [[500,50,15,0.2,feattypes]]

def regression_randomforest_modular(num_train=500,num_test=50,x_range=15,noise_var=0.2,ft=feattypes):
	try:
		from modshogun import RealFeatures, RegressionLabels, CSVFile, RandomForest, MeanRule, PT_REGRESSION
	except ImportError:
		print("Could not import Shogun modules")
		return

	random.seed(1)

	# form training dataset : y=x with noise
	X_train=random.rand(1,num_train)*x_range;
	Y_train=X_train+random.randn(num_train)*noise_var

	# form test dataset
	X_test=array([[float(i)/num_test*x_range for i in range(num_test)]])

	# wrap features and labels into Shogun objects
	feats_train=RealFeatures(X_train)
	feats_test=RealFeatures(X_test)
	train_labels=RegressionLabels(Y_train[0])

	# Random Forest formation
	rand_forest=RandomForest(feats_train,train_labels,20,1)
	rand_forest.set_feature_types(ft)
	rand_forest.set_machine_problem_type(PT_REGRESSION)
	rand_forest.set_combination_rule(MeanRule())
	rand_forest.train()

	# Regress test data
	output=rand_forest.apply_regression(feats_test).get_labels()

	return rand_forest,output

if __name__=='__main__':
	print('RandomForest')
	regression_randomforest_modular(*parameter_list[0])
