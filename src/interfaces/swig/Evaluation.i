/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Sahil Chaddha, Sergey Lisitsyn
 */

/* These functions return new Objects */
%newobject CMachineEvaluation::evaluate();

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::CDirectorContingencyTableEvaluation;
#endif

/* Remove C Prefix */
%rename(Evaluation) CEvaluation;
%rename(EvaluationResult) CEvaluationResult;
%rename(ClusteringAccuracy) CClusteringAccuracy;
%rename(ClusteringMutualInformation) CClusteringMutualInformation;
%rename(DifferentiableFunction) CDifferentiableFunction;
%rename(MachineEvaluation) CMachineEvaluation;
%rename(SplittingStrategy) CSplittingStrategy;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/evaluation/Evaluation.h>
%include <shogun/evaluation/EvaluationResult.h>
%include <shogun/evaluation/ClusteringEvaluation.h>
%include <shogun/evaluation/ClusteringAccuracy.h>
%include <shogun/evaluation/ClusteringMutualInformation.h>
%include <shogun/evaluation/DifferentiableFunction.h>
%include <shogun/evaluation/MachineEvaluation.h>
%include <shogun/evaluation/SplittingStrategy.h>
