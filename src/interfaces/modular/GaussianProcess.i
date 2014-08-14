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
%rename(ConstMean) CConstMean;

%rename(InferenceMethod) CInferenceMethod;
%rename(ExactInferenceMethod) CExactInferenceMethod;
%rename(LaplacianInferenceBase) CLaplacianInferenceBase;
%rename(SingleLaplacianInferenceMethod) CSingleLaplacianInferenceMethod;
%rename(SingleLaplacianInferenceMethodWithLBFGS) CSingleLaplacianInferenceMethodWithLBFGS;
%rename(FITCInferenceMethod) CFITCInferenceMethod;
%rename(EPInferenceMethod) CEPInferenceMethod;

%rename(LikelihoodModel) CLikelihoodModel;
%rename(ProbitLikelihood) CProbitLikelihood;
%rename(LogitLikelihood) CLogitLikelihood;
%rename(GaussianLikelihood) CGaussianLikelihood;
%rename(StudentsTLikelihood) CStudentsTLikelihood;

%rename(VariationalLikelihood) CVariationalLikelihood;
%rename(VariationalGaussianLikelihood) CVariationalGaussianLikelihood;
%rename(NumericalVGLikelihood) CNumericalVGLikelihood;
%rename(DualVariationalGaussianLikelihood) CDualVariationalGaussianLikelihood;
%rename(LogitVGLikelihood) CLogitVGLikelihood;
%rename(LogitVGPiecewiseBoundLikelihood) CLogitVGPiecewiseBoundLikelihood;
%rename(LogitDVGLikelihood) CLogitDVGLikelihood;
%rename(ProbitVGLikelihood) CProbitVGLikelihood;
%rename(StudentsTVGLikelihood) CStudentsTVGLikelihood;

%rename(KLInferenceMethod) CKLInferenceMethod;
%rename(KLLowerTriangularInferenceMethod) CKLLowerTriangularInferenceMethod;
%rename(KLCovarianceInferenceMethod) CKLCovarianceInferenceMethod;
%rename(KLApproxDiagonalInferenceMethod) CKLApproxDiagonalInferenceMethod;
%rename(KLCholeskyInferenceMethod) CKLCholeskyInferenceMethod;
%rename(KLDualInferenceMethod) CKLDualInferenceMethod;

%rename(GaussianProcessMachine) CGaussianProcessMachine;
%rename(GaussianProcessClassification) CGaussianProcessClassification;
%rename(GaussianProcessRegression) CGaussianProcessRegression;

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

%include <shogun/machine/gp/VariationalLikelihood.h>
%include <shogun/machine/gp/VariationalGaussianLikelihood.h>
%include <shogun/machine/gp/NumericalVGLikelihood.h>
%include <shogun/machine/gp/DualVariationalGaussianLikelihood.h>
%include <shogun/machine/gp/LogitVGLikelihood.h>
%include <shogun/machine/gp/LogitVGPiecewiseBoundLikelihood.h>
%include <shogun/machine/gp/LogitDVGLikelihood.h>
%include <shogun/machine/gp/ProbitVGLikelihood.h>
%include <shogun/machine/gp/StudentsTVGLikelihood.h>

%include <shogun/machine/gp/MeanFunction.h>
%include <shogun/machine/gp/ZeroMean.h>
%include <shogun/machine/gp/ConstMean.h>

%include <shogun/machine/gp/InferenceMethod.h>
%include <shogun/machine/gp/LaplacianInferenceBase.h>
%include <shogun/machine/gp/SingleLaplacianInferenceMethod.h>
%include <shogun/machine/gp/ExactInferenceMethod.h>
%include <shogun/machine/gp/SingleLaplacianInferenceMethodWithLBFGS.h>
%include <shogun/machine/gp/FITCInferenceMethod.h>
%include <shogun/machine/gp/EPInferenceMethod.h>

%include <shogun/machine/gp/KLInferenceMethod.h>
%include <shogun/machine/gp/KLLowerTriangularInferenceMethod.h>
%include <shogun/machine/gp/KLCovarianceInferenceMethod.h>
%include <shogun/machine/gp/KLApproxDiagonalInferenceMethod.h>
%include <shogun/machine/gp/KLCholeskyInferenceMethod.h>
%include <shogun/machine/gp/KLDualInferenceMethod.h>

%include <shogun/machine/GaussianProcessMachine.h>
%include <shogun/classifier/GaussianProcessClassification.h>
%include <shogun/regression/GaussianProcessRegression.h>

#endif //HAVE_EIGEN3
