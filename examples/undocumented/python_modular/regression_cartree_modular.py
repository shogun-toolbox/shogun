#!/usr/bin/env python
from numpy import array

# set both input attributes as not nominal (ie. continuous)
feattypes = array([False])

parameter_list = [[500,50,15,0.2,feattypes]]

def regression_cartree_modular(num_train=500,num_test=50,x_range=15,noise_var=0.2,ft=feattypes):	
	try:
		from modshogun import RealFeatures, RegressionLabels, CSVFile, CARTree, PT_REGRESSION
		from numpy import random
	except ImportError:
		print("Could not import Shogun and/or numpy modules")
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

	# CART Tree formation
	c=CARTree(ft,5,PT_REGRESSION,True)
	c.set_labels(train_labels)
	c.train(feats_train)

	# Classify test data
	output=c.apply_regression(feats_test).get_labels()

	return c,output

if __name__=='__main__':
	print('CARTree')
	regression_cartree_modular(*parameter_list[0])
