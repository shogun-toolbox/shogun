/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Sahil Chaddha, Sergey Lisitsyn
 */

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::DirectorContingencyTableEvaluation;
#endif

%shared_ptr(shogun::Evaluation)
%shared_ptr(shogun::EvaluationResult)
%shared_ptr(shogun::GradientResult)
%shared_ptr(shogun::BinaryClassEvaluation)
%shared_ptr(shogun::ClusteringEvaluation)
%shared_ptr(shogun::ClusteringAccuracy)
%shared_ptr(shogun::ClusteringMutualInformation)
%shared_ptr(shogun::ContingencyTableEvaluation)
%shared_ptr(shogun::MachineEvaluation)
%shared_ptr(shogun::MulticlassAccuracy)
%shared_ptr(shogun::MeanAbsoluteError)
%shared_ptr(shogun::MeanSquaredError)
%shared_ptr(shogun::MeanSquaredLogError)
%shared_ptr(shogun::ROCEvaluation)
%shared_ptr(shogun::PRCEvaluation)
%shared_ptr(shogun::AccuracyMeasure)
%shared_ptr(shogun::ErrorRateMeasure)
%shared_ptr(shogun::BALMeasure)
%shared_ptr(shogun::WRACCMeasure)
%shared_ptr(shogun::F1Measure)
%shared_ptr(shogun::CrossCorrelationMeasure)
%shared_ptr(shogun::RecallMeasure)
%shared_ptr(shogun::PrecisionMeasure)
%shared_ptr(shogun::SpecificityMeasure)
%shared_ptr(shogun::SplittingStrategy)
%shared_ptr(shogun::GradientCriterion)
%shared_ptr(shogun::GradientEvaluation)
%shared_ptr(shogun::MulticlassOVREvaluation)
%shared_ptr(shogun::CrossValidationResult)
%shared_ptr(shogun::CrossValidationStorage)
%shared_ptr(shogun::CrossValidationFoldStorage)
%shared_ptr(shogun::StructuredAccuracy)
%shared_ptr(shogun::DirectorContingencyTableEvaluation)
%shared_ptr(shogun::DifferentiableFunction)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/evaluation/EvaluationResult.h>
%include <shogun/evaluation/Evaluation.h>
%include <shogun/evaluation/BinaryClassEvaluation.h>
%include <shogun/evaluation/ClusteringEvaluation.h>
%include <shogun/evaluation/ClusteringAccuracy.h>
%include <shogun/evaluation/ClusteringMutualInformation.h>
%include <shogun/evaluation/ContingencyTableEvaluation.h>
%include <shogun/evaluation/MulticlassAccuracy.h>
%include <shogun/evaluation/MeanAbsoluteError.h>
%include <shogun/evaluation/MeanSquaredError.h>
%include <shogun/evaluation/MeanSquaredLogError.h>
%include <shogun/evaluation/ROCEvaluation.h>
%include <shogun/evaluation/PRCEvaluation.h>
%include <shogun/evaluation/MachineEvaluation.h>
%include <shogun/evaluation/CrossValidationStorage.h>
%include <shogun/evaluation/SplittingStrategy.h>
%include <shogun/evaluation/DifferentiableFunction.h>
%include <shogun/evaluation/GradientCriterion.h>
%include <shogun/evaluation/GradientResult.h>
%include <shogun/evaluation/MulticlassOVREvaluation.h>
%include <shogun/evaluation/StructuredAccuracy.h>
%include <shogun/evaluation/DirectorContingencyTableEvaluation.h>
