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

def evaluation_cross_validation_classification(fm_train=traindat,fm_test=testdat,label_train=label_traindat,\
				       width=2.1,C=1,epsilon=1e-5,tube_epsilon=1e-2):
    from shogun.Evaluation import CrossValidation, CrossValidationResult
    from shogun.Evaluation import MeanSquaredError
    from shogun.Evaluation import StratifiedCrossValidationSplitting
    from shogun.Features import Labels
    from shogun.Features import RealFeatures
    from shogun.Kernel import GaussianKernel
    from shogun.Regression import LibSVR
    from shogun.ModelSelection import GridSearchModelSelection
    from shogun.ModelSelection import ModelSelectionParameters, R_EXP
    from shogun.ModelSelection import ParameterCombination

    # training data
    features_train=RealFeatures(traindat)
    features_test=RealFeatures(testdat)
    labels=Labels(label_traindat)

    # kernel
    kernel=GaussianKernel(features_train, features_train, width)
    labels=Labels(label_train)

    # predictor
    predictor=LibSVR(C, tube_epsilon, kernel, labels)
    predictor.set_epsilon(epsilon)

    # splitting strategy for 5 fold cross-validation (for classification its better
    # to use "StratifiedCrossValidation", but the standard
    # "StratifiedCrossValidationSplitting" is also available
    splitting_strategy=StratifiedCrossValidationSplitting(labels, 5)

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


    # build parameter tree to select C1 and C2 
    param_tree_root=ModelSelectionParameters()
    c1=ModelSelectionParameters("C1");
    param_tree_root.append_child(c1)
    c1.build_values(-2.0, 2.0, R_EXP);

    c2=ModelSelectionParameters("C2");
    param_tree_root.append_child(c2);
    c2.build_values(-2.0, 2.0, R_EXP);

    # model selection instance
    model_selection=GridSearchModelSelection(param_tree_root,
	    cross_validation)

    # perform model selection with selected methods
    #print "performing model selection of"
    print "parameter tree"
    param_tree_root.print_tree()
    best_parameters=model_selection.select_model()

    # print best parameters
    print "best parameters:"
    best_parameters.print_tree()

    # apply them and print result
    best_parameters.apply_to_machine(predictor)
    result=cross_validation.evaluate()
    print "mean:", result.mean
    if result.has_conf_int:
        print "[", result.conf_int_low, ",", result.conf_int_up, "] with alpha=", result.conf_int_alpha

if __name__=='__main__':
	print 'Evaluation CrossValidationClassification'
	evaluation_cross_validation_classification(*parameter_list[0])
