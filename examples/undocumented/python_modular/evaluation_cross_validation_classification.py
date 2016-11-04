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

from numpy.random import randn
from numpy import *

# generate some overlapping training vectors
num_vectors=100
vec_distance=1
traindat=concatenate((randn(2,num_vectors)-vec_distance,
	randn(2,num_vectors)+vec_distance), axis=1)
label_traindat=concatenate((-ones(num_vectors), ones(num_vectors)));

parameter_list = [[traindat,label_traindat]]

def evaluation_cross_validation_classification (traindat=traindat, label_traindat=label_traindat):
    from modshogun import CrossValidation, CrossValidationResult
    from modshogun import ContingencyTableEvaluation, ACCURACY
    from modshogun import StratifiedCrossValidationSplitting
    from modshogun import BinaryLabels
    from modshogun import RealFeatures
    from modshogun import LibLinear, L2R_L2LOSS_SVC

    # training data
    features=RealFeatures(traindat)
    labels=BinaryLabels(label_traindat)

    # classifier
    classifier=LibLinear(L2R_L2LOSS_SVC)

    # splitting strategy for 5 fold cross-validation (for classification its better
    # to use "StratifiedCrossValidation", but the standard
    # "CrossValidationSplitting" is also available
    splitting_strategy=StratifiedCrossValidationSplitting(labels, 5)

    # evaluation method
    evaluation_criterium=ContingencyTableEvaluation(ACCURACY)

    # cross-validation instance
    cross_validation=CrossValidation(classifier, features, labels,
	    splitting_strategy, evaluation_criterium)
    cross_validation.set_autolock(False)

    # (optional) repeat x-val 10 times
    cross_validation.set_num_runs(10)

    # perform cross-validation and print(results)
    result=cross_validation.evaluate()
    #print("mean:", result.mean)

if __name__=='__main__':
	print('Evaluation CrossValidationClassification')
	evaluation_cross_validation_classification(*parameter_list[0])
