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

from numpy.random import randn, seed
from numpy import *

# generate some overlapping training vectors
seed(1)
num_vectors=7
vec_distance=1
traindat=concatenate((randn(2,num_vectors)-vec_distance,
	randn(2,num_vectors)+vec_distance), axis=1)
label_traindat=concatenate((zeros(num_vectors), ones(num_vectors)));

parameter_list = [[traindat,label_traindat]]

def evaluation_cross_validation_multiclass_storage (traindat=traindat, label_traindat=label_traindat):
    from modshogun import CrossValidation, CrossValidationResult
    from modshogun import CrossValidationPrintOutput
    from modshogun import CrossValidationMKLStorage, CrossValidationMulticlassStorage
    from modshogun import MulticlassAccuracy, F1Measure
    from modshogun import StratifiedCrossValidationSplitting
    from modshogun import MulticlassLabels
    from modshogun import RealFeatures, CombinedFeatures
    from modshogun import GaussianKernel, CombinedKernel
    from modshogun import MKLMulticlass
    from modshogun import Statistics, MSG_DEBUG, Math

    Math.init_random(1)

    # training data, combined features all on same data
    features=RealFeatures(traindat)
    comb_features=CombinedFeatures()
    comb_features.append_feature_obj(features)
    comb_features.append_feature_obj(features)
    comb_features.append_feature_obj(features)
    labels=MulticlassLabels(label_traindat)

    # kernel, different Gaussians combined
    kernel=CombinedKernel()
    kernel.append_kernel(GaussianKernel(10, 0.1))
    kernel.append_kernel(GaussianKernel(10, 1))
    kernel.append_kernel(GaussianKernel(10, 2))

    # create mkl using libsvm, due to a mem-bug, interleaved is not possible
    svm=MKLMulticlass(1.0,kernel,labels);
    svm.set_kernel(kernel);

    # splitting strategy for 5 fold cross-validation (for classification its better
    # to use "StratifiedCrossValidation", but the standard
    # "StratifiedCrossValidationSplitting" is also available
    splitting_strategy=StratifiedCrossValidationSplitting(labels, 3)

    # evaluation method
    evaluation_criterium=MulticlassAccuracy()

    # cross-validation instance
    cross_validation=CrossValidation(svm, comb_features, labels,
        splitting_strategy, evaluation_criterium)
    cross_validation.set_autolock(False)

    # append cross vlaidation output classes
    #cross_validation.add_cross_validation_output(CrossValidationPrintOutput())
    #mkl_storage=CrossValidationMKLStorage()
    #cross_validation.add_cross_validation_output(mkl_storage)
    multiclass_storage=CrossValidationMulticlassStorage()
    multiclass_storage.append_binary_evaluation(F1Measure())
    cross_validation.add_cross_validation_output(multiclass_storage)
    cross_validation.set_num_runs(3)

    # perform cross-validation
    result=cross_validation.evaluate()

    roc_0_0_0 = multiclass_storage.get_fold_ROC(0,0,0)
    #print roc_0_0_0
    auc_0_0_0 = multiclass_storage.get_fold_evaluation_result(0,0,0,0)
    #print auc_0_0_0
    return roc_0_0_0, auc_0_0_0


if __name__=='__main__':
	print('Evaluation CrossValidationMulticlassStorage')
	evaluation_cross_validation_multiclass_storage(*parameter_list[0])
