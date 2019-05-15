#!/usr/bin/env python

# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Heiko Strathmann

from numpy.random import randn
from numpy import *

# generate some overlapping training vectors
num_vectors=5
vec_distance=1
traindat=concatenate((randn(2,num_vectors)-vec_distance,
	randn(2,num_vectors)+vec_distance), axis=1)
label_traindat=concatenate((-ones(num_vectors), ones(num_vectors)));

parameter_list = [[traindat,label_traindat]]

def evaluation_cross_validation_mkl_weight_storage(traindat=traindat, label_traindat=label_traindat):
    from shogun import machine_evaluation
    from shogun import parameter_observer
    from shogun import ContingencyTableEvaluation, ACCURACY
    from shogun import splitting_strategy
    from shogun import BinaryLabels
    from shogun import CombinedFeatures
    from shogun import CombinedKernel
    from shogun import MKLClassification
    import shogun as sg
    import numpy as np

    # training data, combined features all on same data
    features=sg.features(traindat)
    comb_features=CombinedFeatures()
    comb_features.append_feature_obj(features)
    comb_features.append_feature_obj(features)
    comb_features.append_feature_obj(features)
    labels=BinaryLabels(label_traindat)

    # kernel, different Gaussians combined
    kernel=CombinedKernel()
    kernel.append_kernel(sg.kernel("GaussianKernel", log_width=np.log(0.1)))
    kernel.append_kernel(sg.kernel("GaussianKernel", log_width=np.log(1)))
    kernel.append_kernel(sg.kernel("GaussianKernel", log_width=np.log(2)))

    # create mkl using libsvm, due to a mem-bug, interleaved is not possible
    libsvm = sg.machine("LibSVM")
    svm = sg.machine("MKLClassification", svm=sg.as_svm(libsvm),
            interleaved_optimization=False, kernel=kernel)

    # splitting strategy for 5 fold cross-validation (for classification its better
    # to use "StratifiedCrossValidation", but the standard
    # "StratifiedCrossValidationSplitting" is also available
    splitting_strategy = splitting_strategy(
        "StratifiedCrossValidationSplitting", labels=labels, num_subsets=5)

    # evaluation method
    evaluation_criterium=ContingencyTableEvaluation(ACCURACY)

    # cross-validation instance
    cross_validation = machine_evaluation(
        "CrossValidation", machine=svm, features=comb_features,
        labels=labels, splitting_strategy=splitting_strategy,
        evaluation_criterion=evaluation_criterium, autolock=False, num_runs=3)

    # append cross vlaidation output classes
    mkl_storage=parameter_observer("ParameterObserverCV")
    cross_validation.subscribe(mkl_storage)

    # perform cross-validation
    result=cross_validation.evaluate()

    # print mkl weights
    weights = []
    for obs_index in range(mkl_storage.get("num_observations")):
        obs = mkl_storage.get_observation(obs_index).get("cross_validation_run")
        for fold_index in range(obs.get("num_folds")):
            fold = obs.get("folds", fold_index)
            w = fold.get("trained_machine").get("kernel").get("combined_kernel_weight")
            weights.append(w)

    print("mkl weights during cross--validation")
    print(weights)

if __name__=='__main__':
	print('Evaluation CrossValidationClassification')
	evaluation_cross_validation_mkl_weight_storage(*parameter_list[0])
