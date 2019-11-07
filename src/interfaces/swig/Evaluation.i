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
%shared_ptr(shogun::ClusteringEvaluation)
%shared_ptr(shogun::ClusteringAccuracy)
%shared_ptr(shogun::ClusteringMutualInformation)
%shared_ptr(shogun::DifferentiableFunction)
%shared_ptr(shogun::MachineEvaluation)
%shared_ptr(shogun::SplittingStrategy)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/evaluation/Evaluation.h>
%include <shogun/evaluation/EvaluationResult.h>
%include <shogun/evaluation/ClusteringEvaluation.h>
%include <shogun/evaluation/ClusteringAccuracy.h>
%include <shogun/evaluation/ClusteringMutualInformation.h>
%include <shogun/evaluation/DifferentiableFunction.h>
%include <shogun/evaluation/MachineEvaluation.h>
%include <shogun/evaluation/SplittingStrategy.h>
