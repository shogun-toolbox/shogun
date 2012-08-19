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
num_vectors=5
vec_distance=1
traindat=concatenate((randn(2,num_vectors)-vec_distance,
	randn(2,num_vectors)+vec_distance), axis=1)
label_traindat=concatenate((-ones(num_vectors), ones(num_vectors)));

parameter_list = [[traindat,label_traindat]]

def evaluation_cross_validation_classification(traindat=traindat, label_traindat=label_traindat):
    from shogun.Evaluation import CrossValidation, CrossValidationResult
    from shogun.Evaluation import CrossValidationPrintOutput
    from shogun.Evaluation import CrossValidationMKLStorage
    from shogun.Evaluation import ContingencyTableEvaluation, ACCURACY
    from shogun.Evaluation import StratifiedCrossValidationSplitting
    from shogun.Features import BinaryLabels
    from shogun.Features import RealFeatures, CombinedFeatures
    from shogun.Kernel import GaussianKernel, CombinedKernel
    from shogun.Classifier import LibSVM, MKLClassification
    from shogun.Mathematics import Statistics

    # training data, combined features all on same data
    features=RealFeatures(traindat)
    comb_features=CombinedFeatures()
    comb_features.append_feature_obj(features)
    comb_features.append_feature_obj(features)
    comb_features.append_feature_obj(features)
    labels=BinaryLabels(label_traindat)
    
    # kernel, different Gaussians combined
    kernel=CombinedKernel()
    kernel.append_kernel(GaussianKernel(10, 0.1))
    kernel.append_kernel(GaussianKernel(10, 1))
    kernel.append_kernel(GaussianKernel(10, 2))

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
    cross_validation.add_cross_validation_output(CrossValidationPrintOutput())
    mkl_storage=CrossValidationMKLStorage()
    cross_validation.add_cross_validation_output(mkl_storage)
    cross_validation.set_num_runs(3)
    
    # perform cross-validation
    result=cross_validation.evaluate()

    # print mkl weights
    weights=mkl_storage.get_mkl_weights()
    print "mkl weights during cross--validation"
    print weights
    
    print "mean per kernel"
    print Statistics.mean(weights, False)
    
    print "variance per kernel"
    print Statistics.variance(weights, False)
    
    print "std-dev per kernel"
    print Statistics.std_deviation(weights, False)
    

if __name__=='__main__':
	print('Evaluation CrossValidationClassification')
	evaluation_cross_validation_classification(*parameter_list[0])
