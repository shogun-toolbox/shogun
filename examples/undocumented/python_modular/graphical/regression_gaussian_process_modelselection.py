#!/usr/bin/env python
from numpy import * 
from pylab import plot, show, legend
import scipy

parameter_list=[[20,100,10,10,0.25,1, 1], [20,100,6,10,0.5,1, 2]]

def regression_gaussian_process_modelselection (n=100,n_test=100, \
		x_range=6,x_range_test=10,noise_var=0.5,width=1, seed=1):
		
	from modshogun import RealFeatures, RegressionLabels
	from modshogun import GaussianKernel
	from modshogun import GradientModelSelection, ModelSelectionParameters, R_LINEAR
	from modshogun import GaussianLikelihood, ZeroMean, \
				ExactInferenceMethod, GaussianProcessRegression, GradientCriterion, \
				GradientEvaluation
		
	# Reproducable results
	random.seed(seed)
	
	# Easy regression data: one dimensional noisy sine wave
	X_train=random.rand(1,n)*x_range
	X_test=array([[float(i)/n_test*x_range_test for i in range(n_test)]])
	Y_test=sin(X_test)
	Y_train=sin(X_train)+random.randn(n)*noise_var
	
	# shogun representation
	labels=RegressionLabels(Y_train[0])
	feats_train=RealFeatures(X_train)
	feats_test=RealFeatures(X_test)
		
	# GP specification
	width=1
	shogun_width=width*width*2
	kernel=GaussianKernel(10,shogun_width)
	kernel.init(feats_train,feats_train)
	zmean = ZeroMean()
	likelihood = GaussianLikelihood()
	inf = ExactInferenceMethod(kernel, feats_train, zmean, labels, likelihood)
	gp = GaussianProcessRegression(inf, feats_train, labels)
	
	# Paramter tree for model selection
	root = ModelSelectionParameters() 
	c1 = ModelSelectionParameters("inference_method", inf)
	root.append_child(c1)

	c2 = ModelSelectionParameters("scale")
	c1.append_child(c2)
	c2.build_values(0.01, 4.0, R_LINEAR)

	c3 = ModelSelectionParameters("likelihood_model", likelihood)
	c1.append_child(c3)

	c4 = ModelSelectionParameters("sigma")
	c3.append_child(c4) 
	c4.build_values(0.001, 4.0, R_LINEAR) 

	c5 = ModelSelectionParameters("kernel", kernel) 
	c1.append_child(c5) 

	c6 = ModelSelectionParameters("width") 
	c5.append_child(c6) 
	c6.build_values(0.001, 4.0, R_LINEAR) 

	# Criterion for Gradient Search
	crit = GradientCriterion()
	
	# Evaluate our inference method for its derivatives
	grad = GradientEvaluation(gp, feats_train, labels, crit)
 
	grad.set_function(inf) 

	gp.print_modsel_params() 

	root.print_tree() 

	# Handles all of the above structures in memory
	grad_search = GradientModelSelection(root, grad) 

	# Set autolocking to false to get rid of warnings	
	grad.set_autolock(False) 

	# Search for best parameters
	best_combination = grad_search.select_model(True)

	# Outputs all result and information
	best_combination.print_tree() 
	best_combination.apply_to_machine(gp)

	result = grad.evaluate()
	result.print_result()
    
	#inference
	gp.set_return_type(GaussianProcessRegression.GP_RETURN_COV) 
	covariance = gp.apply_regression(feats_test) 
	covariance = covariance.get_labels() 
    
	gp.set_return_type(GaussianProcessRegression.GP_RETURN_MEANS) 
	mean = gp.apply_regression(feats_test) 
	mean = mean.get_labels() 

	# some things we can do
	alpha = inf.get_alpha()
	diagonal = inf.get_diagonal_vector()
	cholesky = inf.get_cholesky()
	
	# plot results
	plot(X_train[0],Y_train[0],'x') # training observations
	plot(X_test[0],Y_test[0],'-') # ground truth of test
	plot(X_test[0],mean, '-') # mean predictions of test
	
	legend(["training", "ground truth", "mean predictions"])
	
	show()

	return gp, alpha, labels, diagonal, covariance, mean, cholesky

if __name__=='__main__':
	print('Gaussian Process Regression')
	regression_gaussian_process_modelselection(*parameter_list[1])
