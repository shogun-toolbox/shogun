/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

/* Remove C Prefix */
%rename(Regression) CRegression;
%rename(KernelRidgeRegression) CKernelRidgeRegression;
%rename(LinearRidgeRegression) CLinearRidgeRegression;
%rename(LeastSquaresRegression) CLeastSquaresRegression;
%rename(ExactInferenceMethod) CExactInferenceMethod;
%rename(LaplacianInferenceMethod) CLaplacianInferenceMethod;
%rename(FITCInferenceMethod) CFITCInferenceMethod;
%rename(GaussianLikelihood) CGaussianLikelihood;
%rename(StudentsTLikelihood) CStudentsTLikelihood;
%rename(ZeroMean) CZeroMean;
%rename(GaussianProcessRegression) CGaussianProcessRegression;
%rename(GaussianProcessBinaryClassification) CGaussianProcessBinaryClassification;
%rename(ProbitLikelihood) CProbitLikelihood;
%rename(LeastAngleRegression) CLeastAngleRegression;
%rename(LibSVR) CLibSVR;
%rename(LibLinearRegression) CLibLinearRegression;
%rename(MKL) CMKL;
%rename(MKLRegression) CMKLRegression;
#ifdef USE_SVMLIGHT
%rename(SVRLight) CSVRLight;
#endif //USE_SVMLIGHT

#ifdef HAVE_EIGEN3
%include <shogun/machine/gp/LikelihoodModel.h>
%include <shogun/machine/gp/GaussianLikelihood.h>
%include <shogun/machine/gp/StudentsTLikelihood.h>
%include <shogun/machine/gp/MeanFunction.h>
%include <shogun/machine/gp/ZeroMean.h>
%include <shogun/machine/gp/InferenceMethod.h>
%include <shogun/machine/gp/ExactInferenceMethod.h>
%include <shogun/machine/gp/LaplacianInferenceMethod.h>
%include <shogun/machine/gp/FITCInferenceMethod.h>
%include <shogun/machine/GaussianProcessMachine.h>
%include <shogun/regression/GaussianProcessRegression.h>
%include <shogun/classifier/GaussianProcessBinaryClassification.h>
%include <shogun/machine/gp/ProbitLikelihood.h>

#endif //HAVE_EIGEN3

/* Include Class Headers to make them visible from within the target language */
%include <shogun/regression/Regression.h>
%include <shogun/regression/KernelRidgeRegression.h>
%include <shogun/regression/LinearRidgeRegression.h>
%include <shogun/regression/LeastSquaresRegression.h>
%include <shogun/regression/LeastAngleRegression.h>
%include <shogun/regression/svr/LibSVR.h>
%include <shogun/regression/svr/LibLinearRegression.h>
%include <shogun/classifier/mkl/MKL.h>
%include <shogun/regression/svr/MKLRegression.h>

#ifdef USE_SVMLIGHT
%include <shogun/regression/svr/SVRLight.h>
#endif //USE_SVMLIGHT
