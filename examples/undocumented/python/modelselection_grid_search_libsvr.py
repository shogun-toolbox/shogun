#!/usr/bin/env python

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

def modelselection_grid_search_libsvr (fm_train=traindat,fm_test=testdat,label_train=label_traindat,\
				       width=2.1,C=1,epsilon=1e-5,tube_epsilon=1e-2):
    from shogun import machine_evaluation
    from shogun import splitting_strategy
    from shogun import RegressionLabels
    from shogun import GridSearchModelSelection
    from shogun import ModelSelectionParameters, R_EXP
    import shogun as sg

    # training data
    features_train=sg.features(traindat)
    labels=RegressionLabels(label_traindat)

    # kernel
    kernel=sg.kernel("GaussianKernel", log_width=width)
    kernel.init(features_train, features_train)

    # print all parameter available for modelselection
    # Dont worry if yours is not included but, write to the mailing list
    #kernel.print_modsel_params()

    labels=RegressionLabels(label_train)

    # predictor
    predictor=sg.machine("LibSVR", C1=C, C2=C, tube_epsilon=tube_epsilon,
                         kernel=kernel,
                         labels=labels,
                         epsilon=epsilon)

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

    # build parameter tree to select C1 and C2
    param_tree_root=ModelSelectionParameters()
    c1=ModelSelectionParameters("C1");
    param_tree_root.append_child(c1)
    c1.build_values(-1.0, 0.0, R_EXP);

    c2=ModelSelectionParameters("C2");
    param_tree_root.append_child(c2);
    c2.build_values(-1.0, 0.0, R_EXP);

    # model selection instance
    model_selection=GridSearchModelSelection(cross_validation, param_tree_root)

    # perform model selection with selected methods
    #print "performing model selection of"
    #print "parameter tree"
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

if __name__=='__main__':
	print('ModelselectionGridSearchLibSVR')
	# modelselection_grid_search_libsvr(*parameter_list[0])
