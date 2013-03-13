#!/usr/bin/env python
from numpy import * 
#from pylab import plot, show, legend

parameter_list=[[20,100,6,10,0.5,1], [50,100,10,10,1.5,2]]

def regression_gaussian_process_modular (n=100,n_test=100, \
		x_range=6,x_range_test=10,noise_var=0.5,width=1):
		
	from shogun.Features import RealFeatures, RegressionLabels
	from shogun.Kernel import GaussianKernel
	try:
		from shogun.Regression import GaussianLikelihood, ZeroMean, \
				ExactInferenceMethod, GaussianProcessRegression
	except ImportError:
		print "Eigen3 needed for Gaussian Processes"
		return
	
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
	
	# GP specification
	width=1
	shogun_width=width*width*2
	kernel=GaussianKernel(10, shogun_width)
	zmean = ZeroMean()
	lik = GaussianLikelihood()
	inf = ExactInferenceMethod(kernel, feats_train, zmean, labels, lik)
	gp = GaussianProcessRegression(inf, feats_train, labels)
	
	# some things we can do
	alpha = inf.get_alpha()
	diagonal = inf.get_diagonal_vector()
	cholesky = inf.get_cholesky()
	
	# inference
	gp.set_return_type(GaussianProcessRegression.GP_RETURN_MEANS)
	mean = gp.apply_regression(feats_test)
	gp.set_return_type(GaussianProcessRegression.GP_RETURN_MEANS)
	covariance = gp.apply_regression(feats_test)
	
	# plot results
	#plot(X[0],Y[0],'x') # training observations
	#plot(X_test[0],Y_test[0],'-') # ground truth of test
	#plot(X_test[0],mean.get_labels(), '-') # mean predictions of test
	#legend(["training", "ground truth", "mean predictions"])
	
	#show()

	return gp, alpha, labels, diagonal, covariance, mean, cholesky

if __name__=='__main__':
	print('Gaussian Process Regression')
	regression_gaussian_process_modular(*parameter_list[0])
