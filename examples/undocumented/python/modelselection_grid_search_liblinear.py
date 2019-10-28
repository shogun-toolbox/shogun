#!/usr/bin/env python
#
# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Heiko Strathmann

from numpy.random import randn
from numpy import *

# generate some overlapping training vectors
num_vectors=100
vec_distance=1
traindat=concatenate((randn(2,num_vectors)-vec_distance,
	randn(2,num_vectors)+vec_distance), axis=1)
label_traindat=concatenate((-ones(num_vectors), ones(num_vectors)));

parameter_list = [[traindat,label_traindat]]

def modelselection_grid_search_liblinear (traindat=traindat, label_traindat=label_traindat):
    from shogun import machine_evaluation
    from shogun import splitting_strategy
    from shogun import GridSearchModelSelection
    from shogun import ModelSelectionParameters, R_EXP
    from shogun import BinaryLabels
    import shogun as sg

    # build parameter tree to select C1 and C2
    param_tree_root=ModelSelectionParameters()
    c1=ModelSelectionParameters("C1");
    param_tree_root.append_child(c1)
    c1.build_values(-1.0, 0.0, R_EXP);

    c2=ModelSelectionParameters("C2");
    param_tree_root.append_child(c2);
    c2.build_values(-1.0, 0.0, R_EXP);

    # training data
    features=sg.features(traindat)
    labels=BinaryLabels(label_traindat)

    # classifier
    classifier=sg.machine("LibLinear", liblinear_solver_type="L2R_L2LOSS_SVC")

    # print all parameter available for modelselection
    # Dont worry if yours is not included but, write to the mailing list
    #classifier.print_modsel_params()

    # splitting strategy for cross-validation
    splitting_strategy = splitting_strategy(
        "StratifiedCrossValidationSplitting", labels=labels, num_subsets=10)

    # evaluation method
    evaluation_criterium=sg.evaluation("ContingencyTableEvaluation", type="ACCURACY")

    # cross-validation instance
    cross_validation = machine_evaluation(
        "CrossValidation", machine=classifier, features=features,
        labels=labels, splitting_strategy=splitting_strategy,
        evaluation_criterion=evaluation_criterium)

    # model selection instance
    model_selection=GridSearchModelSelection(cross_validation, param_tree_root)

    # perform model selection with selected methods
    #print "performing model selection of"
    #param_tree_root.print_tree()
    best_parameters=model_selection.select_model()

    # print best parameters
    #print "best parameters:"
    #best_parameters.print_tree()

    # apply them and print result
    best_parameters.apply_to_machine(classifier)
    result=cross_validation.evaluate()
    #result.print_result()

if __name__=='__main__':
    print('ModelSelectionGridSearchLibLinear')
    # modelselection_grid_search_liblinear(*parameter_list[0])
