#!/usr/bin/env python
###########################################################################
# kernel ridge regression
###########################################################################
from numpy import *
#from pylab import plot, show, legend

parameter_list = [[20,100,6,10,0.5,1, 0.5,1], [20,100,6,10,0.5,1, 2,2]]

def regression_kernel_ridge_modular (n=100,n_test=100, \
		x_range=6,x_range_test=10,noise_var=0.5,width=1, tau=1e-6, seed=1):

	from modshogun import RegressionLabels, RealFeatures
	from modshogun import GaussianKernel
	from modshogun import KernelRidgeRegression

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

	krr=KernelRidgeRegression(tau, kernel, labels)
	krr.train(feats_train)

	kernel.init(feats_train, feats_test)
	out = krr.apply().get_labels()

	# plot results
	#plot(X[0],Y[0],'x') # training observations
	#plot(X_test[0],Y_test[0],'-') # ground truth of test
	#plot(X_test[0],out, '-') # mean predictions of test
	#legend(["training", "ground truth", "mean predictions"])
	#show()

	return out,kernel,krr

if __name__=='__main__':
	print('KRR')
	regression_kernel_ridge_modular(*parameter_list[0])
