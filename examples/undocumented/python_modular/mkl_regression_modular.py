#!/usr/bin/env python
from numpy import *

parameter_list=[[20,100,6,10,0.5,1, 1], [20,100,6,10,0.5,1, 2]]

def regression_libsvr_modular (n=100,n_test=100, \
		x_range=6,x_range_test=10,noise_var=0.5,width=1, seed=1):

	from modshogun import RegressionLabels, RealFeatures
	from modshogun import GaussianKernel, PolyKernel, CombinedKernel
	from modshogun import MKLRegression, SVRLight

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

	# combined kernel
	kernel = CombinedKernel()
	kernel.append_kernel(GaussianKernel(10,2))
	kernel.append_kernel(GaussianKernel(10,3))
	kernel.append_kernel(PolyKernel(10,2))
	kernel.init(feats_train, feats_train)

	# constraint generator and MKLRegression
	svr_constraints=SVRLight()
	svr_mkl=MKLRegression(svr_constraints)
	svr_mkl.set_kernel(kernel)
	svr_mkl.set_labels(labels)
	svr_mkl.train()

	# predictions
	kernel.init(feats_train, feats_test)
	out=svr_mkl.apply().get_labels()

	return out, svr_mkl, kernel

if __name__=='__main__':
	print('MKLRegression')
	regression_libsvr_modular(*parameter_list[0])
