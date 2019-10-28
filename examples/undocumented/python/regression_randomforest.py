#!/usr/bin/env python
from numpy import array, random

traindat = '../data/fm_train_real.dat'
testdat = '../data/fm_test_real.dat'
label_traindat = '../data/label_train_multiclass.dat'

# set input attribute as not nominal (ie. continuous)
feattypes = array([False])

parameter_list = [[500,50,15,0.2,feattypes]]

def regression_randomforest(num_train=500,num_test=50,x_range=15,noise_var=0.2,ft=feattypes):
	try:
		from shogun import RegressionLabels, CSVFile
	except ImportError:
		print("Could not import Shogun modules")
		return
	import shogun as sg

	random.seed(1)

	# form training dataset : y=x with noise
	X_train=random.rand(1,num_train)*x_range;
	Y_train=X_train+random.randn(num_train)*noise_var

	# form test dataset
	X_test=array([[float(i)/num_test*x_range for i in range(num_test)]])

	# wrap features and labels into Shogun objects
	feats_train=sg.features(X_train)
	feats_test=sg.features(X_test)
	train_labels=RegressionLabels(Y_train[0])

	# Random Forest formation
	rand_forest=sg.machine("RandomForest", features=feats_train, labels=train_labels, num_bags=20)
	m = rand_forest.get("machine")
	m.put("m_randsubset_size", 1)
	m.put("nominal", ft)	
	rand_forest.put("combination_rule", sg.combination_rule("MeanRule"))
	rand_forest.get_global_parallel().set_num_threads(1)
	rand_forest.train()

	# Regress test data
	output=rand_forest.apply_regression(feats_test).get("labels")

	return rand_forest,output

if __name__=='__main__':
	print('RandomForest')
	regression_randomforest(*parameter_list[0])
