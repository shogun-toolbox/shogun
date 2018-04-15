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
    from shogun import CrossValidation, CrossValidationResult
    from shogun import MeanSquaredError
    from shogun import CrossValidationSplitting
    from shogun import RegressionLabels
    from shogun import RealFeatures
    from shogun import GaussianKernel
    from shogun import LibSVR
    from shogun import GridSearchModelSelection
    from shogun import ModelSelectionParameters, R_EXP
    from shogun import ParameterCombination

    # training data
    features_train=RealFeatures(traindat)
    labels=RegressionLabels(label_traindat)

    # kernel
    kernel=GaussianKernel(features_train, features_train, width)

    # print all parameter available for modelselection
    # Dont worry if yours is not included but, write to the mailing list
    #kernel.print_modsel_params()

    labels=RegressionLabels(label_train)

    # predictor
    predictor=LibSVR(C, tube_epsilon, kernel, labels)
    predictor.set_epsilon(epsilon)

    # splitting strategy for 5 fold cross-validation (for classification its better
    # to use "StratifiedCrossValidation", but the standard
    # "StratifiedCrossValidationSplitting" is also available
    splitting_strategy=CrossValidationSplitting(labels, 5)

    # evaluation method
    evaluation_criterium=MeanSquaredError()

    # cross-validation instance
    cross_validation=CrossValidation(predictor, features_train, labels,
	    splitting_strategy, evaluation_criterium)

#	 (optional) repeat x-val (set larger to get better estimates)
    cross_validation.set_num_runs(2)

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
    # lock data before since model selection will not change the kernel matrix
    # (use with care) This avoids that the kernel matrix is recomputed in every
    # iteration of the model search
    predictor.data_lock(labels, features_train)
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
	modelselection_grid_search_libsvr(*parameter_list[0])
