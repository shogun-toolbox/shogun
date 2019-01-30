/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* Remove C Prefix */
%rename(Regression) CRegression;
%rename(LibSVR) CLibSVR;
%rename(LibLinearRegression) CLibLinearRegression;
%rename(MKL) CMKL;
%rename(MKLRegression) CMKLRegression;

#ifdef USE_SVMLIGHT
%rename(SVRLight) CSVRLight;
#endif //USE_SVMLIGHT


/* Include Class Headers to make them visible from within the target language */
%include <shogun/regression/Regression.h>
%include <shogun/regression/svr/LibSVR.h>
%include <shogun/regression/svr/LibLinearRegression.h>
%include <shogun/classifier/mkl/MKL.h>
%include <shogun/regression/svr/MKLRegression.h>
#ifdef USE_SVMLIGHT
%include <shogun/regression/svr/SVRLight.h>
#endif //USE_SVMLIGHT
