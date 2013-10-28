#!/usr/bin/env python
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2011 Heiko Strathmann
# Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
#

from numpy.random import randn
from numpy import *

# generate some overlapping training vectors
num_vectors=100
vec_distance=1
traindat=concatenate((randn(2,num_vectors)-vec_distance,
	randn(2,num_vectors)+vec_distance), axis=1)
label_traindat=concatenate((-ones(num_vectors), ones(num_vectors)));

parameter_list = [[traindat,label_traindat]]

def modelselection_grid_search_liblinear_modular (traindat=traindat, label_traindat=label_traindat):
    from modshogun import CrossValidation, CrossValidationResult
    from modshogun import ContingencyTableEvaluation, ACCURACY
    from modshogun import StratifiedCrossValidationSplitting
    from modshogun import GridSearchModelSelection
    from modshogun import ModelSelectionParameters, R_EXP
    from modshogun import ParameterCombination
    from modshogun import BinaryLabels
    from modshogun import RealFeatures
    from modshogun import LibLinear, L2R_L2LOSS_SVC

    # build parameter tree to select C1 and C2
    param_tree_root=ModelSelectionParameters()
    c1=ModelSelectionParameters("C1");
    param_tree_root.append_child(c1)
    c1.build_values(-1.0, 0.0, R_EXP);

    c2=ModelSelectionParameters("C2");
    param_tree_root.append_child(c2);
    c2.build_values(-1.0, 0.0, R_EXP);

    # training data
    features=RealFeatures(traindat)
    labels=BinaryLabels(label_traindat)

    # classifier
    classifier=LibLinear(L2R_L2LOSS_SVC)

    # print all parameter available for modelselection
    # Dont worry if yours is not included but, write to the mailing list
    #classifier.print_modsel_params()

    # splitting strategy for cross-validation
    splitting_strategy=StratifiedCrossValidationSplitting(labels, 10)

    # evaluation method
    evaluation_criterium=ContingencyTableEvaluation(ACCURACY)

    # cross-validation instance
    cross_validation=CrossValidation(classifier, features, labels,
                                     splitting_strategy, evaluation_criterium)
    cross_validation.set_autolock(False)

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
    modelselection_grid_search_liblinear_modular(*parameter_list[0])
