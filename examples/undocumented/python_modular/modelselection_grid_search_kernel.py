#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 3 of the License, or
# (at your option) any later version.
#
# Written (C) 2012 Heiko Strathmann
#

from numpy import array
from numpy.random import rand

from shogun.Evaluation import CrossValidation, CrossValidationResult
from shogun.Evaluation import ContingencyTableEvaluation, ACCURACY
from shogun.Evaluation import StratifiedCrossValidationSplitting
from shogun.Features import Labels
from shogun.Features import RealFeatures
from shogun.Kernel import GaussianKernel, PowerKernel
from shogun.Classifier import LibSVM
from shogun.Distance import MinkowskiMetric
from shogun.ModelSelection import GridSearchModelSelection
from shogun.ModelSelection import ModelSelectionParameters, R_EXP, R_LINEAR
from shogun.ModelSelection import ParameterCombination

def create_param_tree():
	root=ModelSelectionParameters()

	c1=ModelSelectionParameters("C1")
	root.append_child(c1)
	c1.build_values(-1.0, 1.0, R_EXP)

	c2=ModelSelectionParameters("C2")
	root.append_child(c2)
	c2.build_values(-1.0, 1.0, R_EXP)

	gaussian_kernel=GaussianKernel()

	# print all parameter available for modelselection
	# Dont worry if yours is not included, simply write to the mailing list
	gaussian_kernel.print_modsel_params()

	param_gaussian_kernel=ModelSelectionParameters("kernel", gaussian_kernel)
	gaussian_kernel_width=ModelSelectionParameters("width")
	gaussian_kernel_width.build_values(-1.0, 1.0, R_EXP, 1.0, 2.0)
	param_gaussian_kernel.append_child(gaussian_kernel_width)
	root.append_child(param_gaussian_kernel)

	power_kernel=PowerKernel()

	# print all parameter available for modelselection
	# Dont worry if yours is not included, simply write to the mailing list
	power_kernel.print_modsel_params()

	param_power_kernel=ModelSelectionParameters("kernel", power_kernel)
	root.append_child(param_power_kernel)

	param_power_kernel_degree=ModelSelectionParameters("degree")
	param_power_kernel_degree.build_values(1.0, 2.0, R_LINEAR)
	param_power_kernel.append_child(param_power_kernel_degree)

	metric=MinkowskiMetric(10)

	# print all parameter available for modelselection
	# Dont worry if yours is not included, simply write to the mailing list
	metric.print_modsel_params()

	param_power_kernel_metric1=ModelSelectionParameters("distance", metric)

	param_power_kernel.append_child(param_power_kernel_metric1)

	param_power_kernel_metric1_k=ModelSelectionParameters("k")
	param_power_kernel_metric1_k.build_values(1.0, 2.0, R_LINEAR)
	param_power_kernel_metric1.append_child(param_power_kernel_metric1_k)

	return root


def modelselection_grid_search_kernel():
	num_subsets=3
	num_vectors=20
	dim_vectors=3

	# create some (non-sense) data
	matrix=rand(dim_vectors, num_vectors)

	# create num_feautres 2-dimensional vectors
	features=RealFeatures()
	features.set_feature_matrix(matrix)

	# create labels, two classes
	labels=Labels(num_vectors)
	for i in range(num_vectors):
		labels.set_label(i, 1 if i%2==0 else -1)

	# create svm
	classifier=LibSVM()

	# splitting strategy
	splitting_strategy=StratifiedCrossValidationSplitting(labels, num_subsets)

	# accuracy evaluation
	evaluation_criterion=ContingencyTableEvaluation(ACCURACY)

	# cross validation class for evaluation in model selection
	cross=CrossValidation(classifier, features, labels, splitting_strategy, evaluation_criterion)
	cross.set_num_runs(1)

	# print all parameter available for modelselection
	# Dont worry if yours is not included, simply write to the mailing list
	classifier.print_modsel_params()

	# model parameter selection
	param_tree=create_param_tree()
	param_tree.print_tree()

	grid_search=GridSearchModelSelection(param_tree, cross)

	print_state=True
	best_combination=grid_search.select_model(print_state)
	print "best parameter(s):"
	best_combination.print_tree()

	best_combination.apply_to_machine(classifier)

	# larger number of runs to have tighter confidence intervals
	cross.set_num_runs(10)
	cross.set_conf_int_alpha(0.01)
	result=cross.evaluate()
	print "result: "
	result.print_result()

	return 0
	
if __name__=='__main__':
	print 'ModelselectionGridSearchKernel'
	modelselection_grid_search_kernel()
