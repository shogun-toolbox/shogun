#!/usr/bin/env python
from numpy import array

# set both input attributes as not nominal (ie. continuous)
feattypes = array([False])

parameter_list = [[50,5,15,0.2,feattypes]]

def regression_cartree(num_train=500,num_test=50,x_range=15,noise_var=0.2,ft=feattypes):
	try:
		from shogun import RegressionLabels, CARTree, PT_REGRESSION
		from numpy import random
	except ImportError:
		print("Could not import Shogun and/or numpy modules")
		return
	import shogun as sg

	random.seed(1)

	# form training dataset : y=x with noise
	X_train=random.rand(1,num_train)*x_range;
	Y_train=X_train+random.randn(num_train)*noise_var

	# form test datasetf
	X_test=array([[float(i)/num_test*x_range for i in range(num_test)]])

	# wrap features and labels into Shogun objects
	feats_train=sg.features(X_train)
	feats_test=sg.features(X_test)
	train_labels=RegressionLabels(Y_train[0])

	# CART Tree formation
	c=CARTree(ft,PT_REGRESSION,5,True)
	c.set_labels(train_labels)
	c.train(feats_train)

	# Classify test data
	output=c.apply_regression(feats_test).get_labels()

	return c,output

if __name__=='__main__':
	print('CARTree')
	regression_cartree(*parameter_list[0])
