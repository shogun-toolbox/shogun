/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

/* Remove C Prefix */
%rename(MeanFunction) CMeanFunction;
%rename(ZeroMean) CZeroMean;
%rename(ConstMean) CConstMean;

%rename(Inference) CInference;
%rename(ExactInferenceMethod) CExactInferenceMethod;
%rename(LaplaceInference) CLaplaceInference;
%rename(SparseInference) CSparseInference;
%rename(SingleSparseInference) CSingleSparseInference;
%rename(SingleFITCInference) CSingleFITCInference;
%rename(SingleLaplaceInferenceMethod) CSingleLaplaceInferenceMethod;
%rename(MultiLaplaceInferenceMethod) CMultiLaplaceInferenceMethod;
%rename(FITCInferenceMethod) CFITCInferenceMethod;
%rename(SingleFITCLaplaceInferenceMethod) CSingleFITCLaplaceInferenceMethod;
%rename(VarDTCInferenceMethod) CVarDTCInferenceMethod;
%rename(EPInferenceMethod) CEPInferenceMethod;

%rename(LikelihoodModel) CLikelihoodModel;
%rename(ProbitLikelihood) CProbitLikelihood;
%rename(LogitLikelihood) CLogitLikelihood;
%rename(SoftMaxLikelihood) CSoftMaxLikelihood;
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

%rename(KLInference) CKLInference;
%rename(KLLowerTriangularInference) CKLLowerTriangularInference;
%rename(KLCovarianceInferenceMethod) CKLCovarianceInferenceMethod;
%rename(KLDiagonalInferenceMethod) CKLDiagonalInferenceMethod;
%rename(KLCholeskyInferenceMethod) CKLCholeskyInferenceMethod;
%rename(KLDualInferenceMethod) CKLDualInferenceMethod;

%rename(KLDualInferenceMethodMinimizer) CKLDualInferenceMethodMinimizer;

%rename(GaussianProcessMachine) CGaussianProcessMachine;
%rename(GaussianProcessClassification) CGaussianProcessClassification;
%rename(GaussianProcessRegression) CGaussianProcessRegression;


/* These functions return new Objects */

/* Include Class Headers to make them visible from within the target language */
%include <shogun/evaluation/DifferentiableFunction.h>
%include <shogun/machine/gp/LikelihoodModel.h>
%include <shogun/machine/gp/ProbitLikelihood.h>
%include <shogun/machine/gp/LogitLikelihood.h>
%include <shogun/machine/gp/SoftMaxLikelihood.h>
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

%include <shogun/machine/gp/Inference.h>
%include <shogun/machine/gp/LaplaceInference.h>
%include <shogun/machine/gp/SparseInference.h>
%include <shogun/machine/gp/SingleSparseInference.h>
%include <shogun/machine/gp/SingleFITCInference.h>
%include <shogun/machine/gp/SingleLaplaceInferenceMethod.h>
%include <shogun/machine/gp/MultiLaplaceInferenceMethod.h>
%include <shogun/machine/gp/ExactInferenceMethod.h>
%include <shogun/machine/gp/SingleFITCLaplaceInferenceMethod.h>
%include <shogun/machine/gp/FITCInferenceMethod.h>
%include <shogun/machine/gp/VarDTCInferenceMethod.h>
%include <shogun/machine/gp/EPInferenceMethod.h>

%include <shogun/machine/gp/KLInference.h>
%include <shogun/machine/gp/KLLowerTriangularInference.h>
%include <shogun/machine/gp/KLCovarianceInferenceMethod.h>
%include <shogun/machine/gp/KLDiagonalInferenceMethod.h>
%include <shogun/machine/gp/KLCholeskyInferenceMethod.h>
%include <shogun/machine/gp/KLDualInferenceMethod.h>

%include <shogun/machine/GaussianProcessMachine.h>
%include <shogun/classifier/GaussianProcessClassification.h>
%include <shogun/regression/GaussianProcessRegression.h>
