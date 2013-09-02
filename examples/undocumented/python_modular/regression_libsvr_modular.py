#!/usr/bin/env python
from numpy import * 

parameter_list=[[1, 0.1,20,100,6,10,0.5,1, 1], [2,0.2,20,100,6,10,0.5,1, 2]]

def regression_libsvr_modular (svm_c=1, svr_param=0.1, n=100,n_test=100, \
		x_range=6,x_range_test=10,noise_var=0.5,width=1, seed=1):

	from modshogun import RegressionLabels, RealFeatures
	from modshogun import GaussianKernel
	from modshogun import LibSVR, LIBSVR_NU_SVR, LIBSVR_EPSILON_SVR

	# reproducable results
	random.seed(seed)
	
	# easy regression data: one dimensional noisy sine wave
	n=15
	n_test=100
	x_range_test=10
	noise_var=0.5;
	X=random.rand(1,n)*x_range
	
	X_test=array([[float(i)/n_test*x_range_test for i in range(n_test)]])
	Y_test=sin(X_test)
	Y=sin(X)+random.randn(n)*noise_var
	
	# shogun representation
	labels=RegressionLabels(Y[0])
	feats_train=RealFeatures(X)
	feats_test=RealFeatures(X_test)

	kernel=GaussianKernel(feats_train, feats_train, width)
	
	# two svr models: epsilon and nu
	svr_epsilon=LibSVR(svm_c, svr_param, kernel, labels, LIBSVR_EPSILON_SVR)
	svr_epsilon.train()
	svr_nu=LibSVR(svm_c, svr_param, kernel, labels, LIBSVR_NU_SVR)
	svr_nu.train()

	# predictions
	kernel.init(feats_train, feats_test)
	out1_epsilon=svr_epsilon.apply().get_labels()
	out2_epsilon=svr_epsilon.apply(feats_test).get_labels()
	out1_nu=svr_epsilon.apply().get_labels()
	out2_nu=svr_epsilon.apply(feats_test).get_labels()

	return out1_epsilon,out2_epsilon,out1_nu,out2_nu ,kernel

if __name__=='__main__':
	print('LibSVR')
	regression_libsvr_modular(*parameter_list[0])
