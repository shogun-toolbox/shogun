#!/usr/bin/env python

from numpy import *
from pylab import plot, show, legend, fill_between, figure, subplot, title

def regression_gaussian_process_modelselection (n=100, n_test=100, \
		x_range=5, x_range_test=10, noise_var=0.4):

	from modshogun import RealFeatures, RegressionLabels
	from modshogun import GaussianKernel
	from modshogun import GradientModelSelection, ModelSelectionParameters
	from modshogun import GaussianLikelihood, ZeroMean, \
		ExactInferenceMethod, GaussianProcessRegression, GradientCriterion, \
		GradientEvaluation

	# easy regression data: one dimensional noisy sine wave
	X_train = random.rand(1,n)*x_range
	X_test = array([[float(i)/n_test*x_range_test for i in range(n_test)]])

	y_test = sin(X_test)
	y_train = sin(X_train)+random.randn(n)*noise_var

	# shogun representation
	labels = RegressionLabels(y_train[0])
	feats_train = RealFeatures(X_train)
	feats_test = RealFeatures(X_test)

	# GP specification
	kernel = GaussianKernel(10, 0.05)

	mean = ZeroMean()

	likelihood = GaussianLikelihood(0.8)

	inf = ExactInferenceMethod(kernel, feats_train, mean, labels, likelihood)
	inf.set_scale(2.5)

	gp = GaussianProcessRegression(inf)

	means = gp.get_mean_vector(feats_test)
	variances = gp.get_variance_vector(feats_test)

	# plot results
	figure()

	subplot(2, 1, 1)
	title('Initial parameter\'s values')

	plot(X_train[0], y_train[0], 'bx') # training observations

	plot(X_test[0], y_test[0], 'g-') # ground truth of test
	plot(X_test[0], means, 'r-') # mean predictions of test

	fill_between(X_test[0], means-1.96*sqrt(variances),
				 means+1.96*sqrt(variances), color='grey')

	legend(["training", "ground truth", "mean predictions"])

	# evaluate our inference method for its derivatives
	grad = GradientEvaluation(gp, feats_train, labels, GradientCriterion(), False)
	grad.set_function(inf)

	# handles all of the above structures in memory
	grad_search = GradientModelSelection(grad)

	# search for best parameters
	best_combination = grad_search.select_model(True)

	# outputs all result and information
	best_combination.apply_to_machine(gp)

	means = gp.get_mean_vector(feats_test)
	variances = gp.get_variance_vector(feats_test)

	# plot results
	subplot(2, 1, 2)
	title('Selected by gradient search parameter\'s values')

	plot(X_train[0], y_train[0], 'bx') # training observations

	plot(X_test[0], y_test[0], 'g-') # ground truth of test
	plot(X_test[0], means, 'r-') # mean predictions of test

	fill_between(X_test[0], means-1.96*sqrt(variances),
				 means+1.96*sqrt(variances), color='grey')

	legend(["training", "ground truth", "mean predictions"])

	show()

if __name__=='__main__':

	regression_gaussian_process_modelselection()
