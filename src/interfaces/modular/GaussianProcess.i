/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

/* Remove C Prefix */
#ifdef HAVE_EIGEN3
%rename(MeanFunction) CMeanFunction;
%rename(ZeroMean) CZeroMean;

%rename(InferenceMethod) CInferenceMethod;
%rename(ExactInferenceMethod) CExactInferenceMethod;
%rename(LaplacianInferenceMethod) CLaplacianInferenceMethod;
%rename(FITCInferenceMethod) CFITCInferenceMethod;

%rename(LikelihoodModel) CLikelihoodModel;
%rename(ProbitLikelihood) CProbitLikelihood;
%rename(LogitLikelihood) CLogitLikelihood;
%rename(GaussianLikelihood) CGaussianLikelihood;
%rename(StudentsTLikelihood) CStudentsTLikelihood;

%rename(GaussianProcessMachine) CGaussianProcessMachine;
%rename(GaussianProcessBinaryClassification) CGaussianProcessBinaryClassification;
%rename(GaussianProcessBinaryRegression) CGaussianProcessBinaryRegression;

#endif //HAVE_EIGEN3

/* These functions return new Objects */

/* Include Class Headers to make them visible from within the target language */
#ifdef HAVE_EIGEN3
%include <shogun/evaluation/DifferentiableFunction.h>
%include <shogun/machine/gp/LikelihoodModel.h>
%include <shogun/machine/gp/ProbitLikelihood.h>
%include <shogun/machine/gp/LogitLikelihood.h>
%include <shogun/machine/gp/GaussianLikelihood.h>
%include <shogun/machine/gp/StudentsTLikelihood.h>
 
%include <shogun/machine/gp/MeanFunction.h>
%include <shogun/machine/gp/ZeroMean.h>
 
%include <shogun/machine/gp/InferenceMethod.h>
%include <shogun/machine/gp/LaplacianInferenceMethod.h>
%include <shogun/machine/gp/ExactInferenceMethod.h>
%include <shogun/machine/gp/LaplacianInferenceMethod.h>
%include <shogun/machine/gp/FITCInferenceMethod.h>

%include <shogun/machine/GaussianProcessMachine.h>
%include <shogun/classifier/GaussianProcessBinaryClassification.h>
%include <shogun/regression/GaussianProcessRegression.h>
 
%include <shogun/machine/gp/MeanFunction.h>
%include <shogun/machine/gp/ZeroMean.h>
#endif //HAVE_EIGEN3

