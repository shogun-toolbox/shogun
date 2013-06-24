#!/usr/bin/env python
from numpy import random,array,sin,round#,sqrt
#from pylab import plot, show, legend, fill_between

parameter_list=[[20,100,6,10,0.05,1, 1], [10,30,7,9,0.5,0.5, 2]]

def regression_gaussian_process_modular (n=100,n_test=100, \
		x_range=6,x_range_test=10,noise_var=0.5,width=1, seed=1):

	from modshogun import RealFeatures, RegressionLabels, GaussianKernel, Math
	try:
		from modshogun import GaussianLikelihood, ZeroMean, \
				ExactInferenceMethod, GaussianProcessRegression
	except ImportError:
		print("Eigen3 needed for Gaussian Processes")
		return

	# reproducable results
	random.seed(seed)
	Math.init_random(17)

	# easy regression data: one dimensional noisy sine wave
	X=random.rand(1,n)*x_range

	X_test=array([[float(i)/n_test*x_range_test for i in range(n_test)]])
	Y_test=sin(X_test)
	Y=sin(X)+random.randn(n)*noise_var

	# shogun representation
	labels=RegressionLabels(Y[0])
	feats_train=RealFeatures(X)
	feats_test=RealFeatures(X_test)

	# GP specification
	shogun_width=width*width*2
	kernel=GaussianKernel(10, shogun_width)
	zmean = ZeroMean()
	lik = GaussianLikelihood()
	lik.set_sigma(noise_var)
	inf = ExactInferenceMethod(kernel, feats_train, zmean, labels, lik)

	# train GP
	gp = GaussianProcessRegression(inf)
	gp.train()

	# some things we can do
	alpha = inf.get_alpha()
	diagonal = inf.get_diagonal_vector()
	cholesky = inf.get_cholesky()

	# get mean and variance vectors
	mean = gp.get_mean_vector(feats_test)
	variance = gp.get_variance_vector(feats_test)

	# plot results
	#plot(X[0],Y[0],'x') # training observations
	#plot(X_test[0],Y_test[0],'-') # ground truth of test
	#plot(X_test[0],mean, '-') # mean predictions of test
	#fill_between(X_test[0],mean-1.96*sqrt(variance),mean+1.96*sqrt(variance),color='grey')  # 95% confidence interval
	#legend(["training", "ground truth", "mean predictions"])

	#show()

	return alpha, diagonal, round(variance,12), round(mean,12), cholesky

if __name__=='__main__':
	print('Gaussian Process Regression')
	regression_gaussian_process_modular(*parameter_list[0])
