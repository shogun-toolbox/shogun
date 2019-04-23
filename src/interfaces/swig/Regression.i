/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* Remove C Prefix */
%shared_ptr(shogun::Regression)
%shared_ptr(shogun::MKL)
%shared_ptr(shogun::MKLRegression)

#ifdef USE_SVMLIGHT
%shared_ptr(shogun::SVRLight)
#endif //USE_SVMLIGHT


/* Include Class Headers to make them visible from within the target language */
%include <shogun/regression/Regression.h>
%include <shogun/classifier/mkl/MKL.h>
%include <shogun/regression/svr/MKLRegression.h>
#ifdef USE_SVMLIGHT
%include <shogun/regression/svr/SVRLight.h>
#endif //USE_SVMLIGHT
