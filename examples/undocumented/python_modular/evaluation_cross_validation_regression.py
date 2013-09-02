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

parameter_list = [[traindat,testdat,label_traindat,0.8,1e-6],[traindat,testdat,label_traindat,0.9,1e-7]]

def evaluation_cross_validation_regression (fm_train=traindat,fm_test=testdat,label_train=label_traindat,width=0.8,tau=1e-6):
    from modshogun import CrossValidation, CrossValidationResult
    from modshogun import MeanSquaredError
    from modshogun import CrossValidationSplitting
    from modshogun import RegressionLabels, RealFeatures
    from modshogun import GaussianKernel
    from modshogun import KernelRidgeRegression

    # training data
    features=RealFeatures(fm_train)
    labels=RegressionLabels(label_train)

    # kernel and predictor
    kernel=GaussianKernel()
    predictor=KernelRidgeRegression(tau, kernel, labels)

    # splitting strategy for 5 fold cross-validation (for classification its better
    # to use "StratifiedCrossValidation", but here, the std x-val is used
    splitting_strategy=CrossValidationSplitting(labels, 5)

    # evaluation method
    evaluation_criterium=MeanSquaredError()

    # cross-validation instance
    cross_validation=CrossValidation(predictor, features, labels,
        splitting_strategy, evaluation_criterium)

    # (optional) repeat x-val 10 times
    cross_validation.set_num_runs(10)

    # (optional) request 95% confidence intervals for results (not actually needed
    # for this toy example)
    cross_validation.set_conf_int_alpha(0.05)

    # (optional) tell machine to precompute kernel matrix. speeds up. may not work
    predictor.data_lock(labels, features)

    # perform cross-validation and print(results)
    result=cross_validation.evaluate()
    #print("mean:", result.mean)
    #if result.has_conf_int:
    #    print("[", result.conf_int_low, ",", result.conf_int_up, "] with alpha=", result.conf_int_alpha)

if __name__=='__main__':
	print('Evaluation CrossValidationClassification')
	evaluation_cross_validation_regression(*parameter_list[0])
