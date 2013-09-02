#!/usr/bin/env python
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2012 Heiko Strathmann
# Copyright (C) 2012 Berlin Institute of Technology and Max-Planck-Society
#

from numpy import array
from numpy.random import seed, rand
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')


parameter_list = [[traindat,testdat,label_traindat,2.1,1,1e-5,1e-2], \
                 [traindat,testdat,label_traindat,2.1,1,1e-5,1e-2]]

def modelselection_grid_search_krr_modular (fm_train=traindat,fm_test=testdat,label_train=label_traindat,\
				       width=2.1,C=1,epsilon=1e-5,tube_epsilon=1e-2):
    from modshogun import CrossValidation, CrossValidationResult
    from modshogun import MeanSquaredError
    from modshogun import CrossValidationSplitting
    from modshogun import RegressionLabels
    from modshogun import RealFeatures
    from modshogun import KernelRidgeRegression
    from modshogun import GridSearchModelSelection
    from modshogun import ModelSelectionParameters

    # training data
    features_train=RealFeatures(traindat)
    features_test=RealFeatures(testdat)
    labels=RegressionLabels(label_traindat)

    # labels
    labels=RegressionLabels(label_train)

    # predictor, set tau=0 here, doesnt matter
    predictor=KernelRidgeRegression()

    # splitting strategy for 5 fold cross-validation (for classification its better
    # to use "StratifiedCrossValidation", but the standard
    # "StratifiedCrossValidationSplitting" is also available
    splitting_strategy=CrossValidationSplitting(labels, 5)

    # evaluation method
    evaluation_criterium=MeanSquaredError()

    # cross-validation instance
    cross_validation=CrossValidation(predictor, features_train, labels,
	    splitting_strategy, evaluation_criterium)
	
    # (optional) repeat x-val 10 times
    cross_validation.set_num_runs(10)

    # (optional) request 95% confidence intervals for results (not actually needed
    # for this toy example)
    cross_validation.set_conf_int_alpha(0.05)

    # print all parameter available for modelselection
    # Dont worry if yours is not included but, write to the mailing list
    #predictor.print_modsel_params()

    # build parameter tree to select regularization parameter
    param_tree_root=create_param_tree()

    # model selection instance
    model_selection=GridSearchModelSelection(param_tree_root,
	    cross_validation)

    # perform model selection with selected methods
    #print "performing model selection of"
    #print "parameter tree:"
    #param_tree_root.print_tree()
    
    #print "starting model selection"
    # print the current parameter combination, if no parameter nothing is printed
    print_state=False
    
    best_parameters=model_selection.select_model(print_state)

    # print best parameters
    #print "best parameters:"
    #best_parameters.print_tree()

    # apply them and print result
    best_parameters.apply_to_machine(predictor)
    result=cross_validation.evaluate()
    #print "mean:", result.mean
    #if result.has_conf_int:
    #    print "[", result.conf_int_low, ",", result.conf_int_up, "] with alpha=", result.conf_int_alpha

# creates all the parameters to optimize
def create_param_tree():
    from modshogun import ModelSelectionParameters, R_EXP, R_LINEAR
    from modshogun import ParameterCombination
    from modshogun import GaussianKernel, PolyKernel
    root=ModelSelectionParameters()

    tau=ModelSelectionParameters("tau")
    root.append_child(tau)

    # also R_LINEAR/R_LOG is available as type
    min=-1
    max=1
    type=R_EXP
    step=1.5
    base=2
    tau.build_values(min, max, type, step, base)

    # gaussian kernel with width
    gaussian_kernel=GaussianKernel()
    
    # print all parameter available for modelselection
    # Dont worry if yours is not included but, write to the mailing list
    #gaussian_kernel.print_modsel_params()
    
    param_gaussian_kernel=ModelSelectionParameters("kernel", gaussian_kernel)
    gaussian_kernel_width=ModelSelectionParameters("width");
    gaussian_kernel_width.build_values(5.0, 8.0, R_EXP, 1.0, 2.0)
    param_gaussian_kernel.append_child(gaussian_kernel_width)
    root.append_child(param_gaussian_kernel)

    # polynomial kernel with degree
    poly_kernel=PolyKernel()
    
    # print all parameter available for modelselection
    # Dont worry if yours is not included but, write to the mailing list
    #poly_kernel.print_modsel_params()
    
    param_poly_kernel=ModelSelectionParameters("kernel", poly_kernel)

    root.append_child(param_poly_kernel)

    # note that integers are used here
    param_poly_kernel_degree=ModelSelectionParameters("degree")
    param_poly_kernel_degree.build_values(1, 2, R_LINEAR)
    param_poly_kernel.append_child(param_poly_kernel_degree)

    return root


if __name__=='__main__':
	print('ModelselectionGridSearchKRR')
	modelselection_grid_search_krr_modular(*parameter_list[0])
