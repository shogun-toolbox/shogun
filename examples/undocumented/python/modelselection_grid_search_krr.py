#!/usr/bin/env python
#
# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Heiko Strathmann

from numpy import array
from numpy.random import seed, rand
from tools.load import LoadMatrix
lm=LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')
label_traindat = lm.load_labels('../data/label_train_twoclass.dat')


parameter_list = [[traindat,testdat,label_traindat,2.1,1,1e-5,1e-2], \
                 [traindat,testdat,label_traindat,2.1,1,1e-5,1e-2]]

def modelselection_grid_search_krr (fm_train=traindat,fm_test=testdat,label_train=label_traindat,\
				       width=2.1,C=1,epsilon=1e-5,tube_epsilon=1e-2):
    from shogun import machine_evaluation, splitting_strategy
    from shogun import RegressionLabels
    from shogun import GridSearchModelSelection
    import shogun as sg

    # training data
    features_train=sg.features(traindat)
    features_test=sg.features(testdat)
    labels=RegressionLabels(label_traindat)

    # labels
    labels=RegressionLabels(label_train)

    # predictor, set tau=0 here, doesnt matter
    predictor=sg.machine("KernelRidgeRegression")

    # splitting strategy for 5 fold cross-validation (for classification its better
    # to use "StratifiedCrossValidation", but the standard
    # "StratifiedCrossValidationSplitting" is also available
    splitting_strategy = splitting_strategy(
        "CrossValidationSplitting", labels=labels, num_subsets=5)

    # evaluation method
    evaluation_criterium=sg.evaluation("MeanSquaredError")

    # cross-validation instance
    cross_validation = machine_evaluation(
        "CrossValidation", machine=predictor, features=features_train,
        labels=labels, splitting_strategy=splitting_strategy,
        evaluation_criterion=evaluation_criterium, num_runs=2)

    # print all parameter available for modelselection
    # Dont worry if yours is not included but, write to the mailing list
    #predictor.print_modsel_params()

    # build parameter tree to select regularization parameter
    param_tree_root=create_param_tree()

    # model selection instance
    model_selection=GridSearchModelSelection(cross_validation, param_tree_root)

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

# creates all the parameters to optimize
def create_param_tree():
    from shogun import ModelSelectionParameters, R_EXP, R_LINEAR
    import math
    import shogun as sg
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
    gaussian_kernel=sg.kernel("GaussianKernel")

    # print all parameter available for modelselection
    # Dont worry if yours is not included but, write to the mailing list
    #gaussian_kernel.print_modsel_params()

    param_gaussian_kernel=ModelSelectionParameters("kernel", gaussian_kernel)
    gaussian_kernel_width=ModelSelectionParameters("log_width");
    gaussian_kernel_width.build_values(2.0*math.log(2.0), 2.5*math.log(2.0), R_LINEAR, 1.0)
    param_gaussian_kernel.append_child(gaussian_kernel_width)
    root.append_child(param_gaussian_kernel)

    # polynomial kernel with degree
    poly_kernel=sg.kernel("PolyKernel")

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
	# modelselection_grid_search_krr(*parameter_list[0])
