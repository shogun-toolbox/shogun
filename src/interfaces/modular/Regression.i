/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Written (W) 2013 Heiko Strathmann
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

/* Remove C Prefix */
%rename(Regression) CRegression;
%rename(KernelRidgeRegression) CKernelRidgeRegression;
%rename(LinearRidgeRegression) CLinearRidgeRegression;
%rename(LeastSquaresRegression) CLeastSquaresRegression;
%rename(LeastAngleRegression) CLeastAngleRegression;
%rename(LibSVR) CLibSVR;
%rename(LibLinearRegression) CLibLinearRegression;
%rename(MKL) CMKL;
%rename(MKLRegression) CMKLRegression;

#ifdef USE_SVMLIGHT
%rename(SVRLight) CSVRLight;
#endif //USE_SVMLIGHT


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
