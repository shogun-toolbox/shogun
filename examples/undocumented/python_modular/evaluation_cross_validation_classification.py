#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2012 Heiko Strathmann
# Copyright (C) 2012 Berlin Institute of Technology and Max-Planck-Society
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

def evaluation_cross_validation_classification(traindat=traindat, label_traindat=label_traindat):
    from shogun.Evaluation import CrossValidation, CrossValidationResult
    from shogun.Evaluation import ContingencyTableEvaluation, ACCURACY
    from shogun.Evaluation import StratifiedCrossValidationSplitting
    from shogun.ModelSelection import GridSearchModelSelection
    from shogun.ModelSelection import ModelSelectionParameters, R_EXP
    from shogun.ModelSelection import ParameterCombination
    from shogun.Features import Labels
    from shogun.Features import RealFeatures
    from shogun.Classifier import LibLinear, L2R_L2LOSS_SVC

    # build parameter tree to select C1 and C2 
    param_tree_root=ModelSelectionParameters()
    c1=ModelSelectionParameters("C1");
    param_tree_root.append_child(c1)
    c1.build_values(-2.0, 2.0, R_EXP);

    c2=ModelSelectionParameters("C2");
    param_tree_root.append_child(c2);
    c2.build_values(-2.0, 2.0, R_EXP);

    # training data
    features=RealFeatures(traindat)
    labels=Labels(label_traindat)

    # classifier
    classifier=LibLinear(L2R_L2LOSS_SVC)

    # splitting strategy for 5 fold cross-validation (for classification its better
    # to use "StratifiedCrossValidation", but the standard
    # "StratifiedCrossValidationSplitting" is also available
    splitting_strategy=StratifiedCrossValidationSplitting(labels, 5)

    # evaluation method
    evaluation_criterium=ContingencyTableEvaluation(ACCURACY)

    # cross-validation instance
    cross_validation=CrossValidation(classifier, features, labels,
	    splitting_strategy, evaluation_criterium)
	
    # (optional) repeat x-val 10 times
    cross_validation.set_num_runs(10)

    # (optional) request 95% confidence intervals for results (not actually needed
    # for this toy example)
    cross_validation.set_conf_int_alpha(0.05)
	
    # perform cross-validation and print results
    result=cross_validation.evaluate()
    result.print_result()

if __name__=='__main__':
	print 'Evaluation CrossValidationClassification'
	modelselection_grid_search_linear_modular(*parameter_list[0])
