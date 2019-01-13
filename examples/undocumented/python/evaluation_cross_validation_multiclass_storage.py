#!/usr/bin/env python
#
# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Heiko Strathmann

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
    from shogun import CrossValidation, CrossValidationResult
    from shogun import ParameterObserverCV
    from shogun import MulticlassAccuracy, F1Measure
    from shogun import StratifiedCrossValidationSplitting
    from shogun import MulticlassLabels
    from shogun import RealFeatures, CombinedFeatures
    from shogun import CombinedKernel
    from shogun import MKLMulticlass
    from shogun import Statistics, MSG_DEBUG, Math
    from shogun import ROCEvaluation
    import shogun as sg
    import numpy as np

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
    kernel.append_kernel(sg.kernel("GaussianKernel", log_width=np.log(0.1)))
    kernel.append_kernel(sg.kernel("GaussianKernel", log_width=np.log(1)))
    kernel.append_kernel(sg.kernel("GaussianKernel", log_width=np.log(2)))

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

    # append cross validation parameter observer
    multiclass_storage=ParameterObserverCV()
    cross_validation.subscribe_to_parameters(multiclass_storage)
    cross_validation.set_num_runs(3)

    # perform cross-validation
    result=cross_validation.evaluate()

    # get first observation and first fold
    obs = multiclass_storage.get_observations()[0]
    fold = obs.get_folds_results()[0]

    # get fold ROC for first class
    eval_ROC = ROCEvaluation()
    pred_lab_binary = MulticlassLabels.obtain_from_generic(fold.get_test_result()).get_binary_for_class(0)
    true_lab_binary = MulticlassLabels.obtain_from_generic(fold.get_test_true_result()).get_binary_for_class(0)
    eval_ROC.evaluate(pred_lab_binary, true_lab_binary)
    print eval_ROC.get_ROC()

    # get fold evaluation result
    acc_measure = F1Measure()
    print acc_measure.evaluate(pred_lab_binary, true_lab_binary)


if __name__=='__main__':
	print('Evaluation CrossValidationMulticlassStorage')
	evaluation_cross_validation_multiclass_storage(*parameter_list[0])
