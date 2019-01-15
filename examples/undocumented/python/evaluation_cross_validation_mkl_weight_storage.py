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
    from shogun import CrossValidation, CrossValidationResult
    from shogun import ParameterObserverCV
    from shogun import ContingencyTableEvaluation, ACCURACY
    from shogun import StratifiedCrossValidationSplitting
    from shogun import BinaryLabels
    from shogun import RealFeatures, CombinedFeatures
    from shogun import CombinedKernel
    from shogun import LibSVM, MKLClassification
    import shogun as sg
    import numpy as np

    # training data, combined features all on same data
    features=RealFeatures(traindat)
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
    svm=MKLClassification(LibSVM());
    svm.set_interleaved_optimization_enabled(False);
    svm.set_kernel(kernel);

    # splitting strategy for 5 fold cross-validation (for classification its better
    # to use "StratifiedCrossValidation", but the standard
    # "StratifiedCrossValidationSplitting" is also available
    splitting_strategy=StratifiedCrossValidationSplitting(labels, 5)

    # evaluation method
    evaluation_criterium=ContingencyTableEvaluation(ACCURACY)

    # cross-validation instance
    cross_validation=CrossValidation(svm, comb_features, labels,
        splitting_strategy, evaluation_criterium)
    cross_validation.set_autolock(False)

    # append cross vlaidation output classes
    mkl_storage=ParameterObserverCV()
    cross_validation.subscribe_to_parameters(mkl_storage)
    cross_validation.set_num_runs(3)

    # perform cross-validation
    result=cross_validation.evaluate()

    # print mkl weights
    weights = []
    for obs_index in range(mkl_storage.get_num_observations()):
        obs = mkl_storage.get_observation(obs_index)
        for fold_index in range(obs.get_num_folds()):
            fold = obs.get_fold(fold_index)
            machine = MKLClassification.obtain_from_generic(fold.get_trained_machine())
            w = machine.get_kernel().get_subkernel_weights()
            weights.append(w)

    print("mkl weights during cross--validation")
    print(weights)

if __name__=='__main__':
	print('Evaluation CrossValidationClassification')
	evaluation_cross_validation_mkl_weight_storage(*parameter_list[0])
